from __future__ import annotations
import contextlib
import dis
import functools
import inspect
import logging
import os
import sys
import textwrap
import threading
import traceback
import types
import warnings
from dataclasses import dataclass
from enum import Enum
from os.path import dirname, join
from typing import (
from unittest.mock import patch
import torch
import torch.fx
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch import _guards
from torch._subclasses import fake_tensor
from torch.export import Constraint
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
from torch.fx.experimental.symbolic_shapes import (
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.nn.parallel.distributed import DistributedDataParallel
from ..fx import GraphModule
from .backends.registry import CompilerFn, lookup_backend
from .hooks import Hooks
from . import config, convert_frame, external_utils, skipfiles, utils
from .code_context import code_context
from .exc import CondOpArgsMismatchError, UserError, UserErrorType
from .mutation_guard import install_generation_tagging_init
from .types import CacheEntry, DynamoCallback
from .utils import compile_times
from torch._dispatch.python import enable_python_dispatcher
from torch.utils._python_dispatch import _disable_current_modes
import sympy
def rewrite_signature(f_sig, graph, fake_mode, flat_args, in_spec, example_fake_inputs, graph_captured_input, graph_captured_output, dynamo_traced_result, flat_args_dynamic_dims):
    orig_args, orig_kwargs = pytree.tree_unflatten(flat_args, in_spec)
    supported_types = (torch.Tensor, torch.SymInt, torch.SymFloat, torch.SymBool)

    def is_supported_type(val):
        return isinstance(val, supported_types)

    def produce_matching(sources, candidates):
        source_types = ' or '.join([desc + ' of types: (' + ', '.join([str(type(val)) for val in vals]) + ')' for desc, vals in sources.items()])
        source_vals = [val for vals in sources.values() for val in vals]
        matched_elements_positions = []
        dict_of_source_vals = {}
        for i, val in enumerate(source_vals):
            dict_of_source_vals[id(val)] = i
        for candidate_desc, candidate_vals in candidates.items():
            for i, val in enumerate(candidate_vals):
                if is_supported_type(val):
                    if id(val) in dict_of_source_vals:
                        matched_elements_positions.append(dict_of_source_vals[id(val)])
                    else:
                        raise AssertionError(f'{candidate_desc} #{i + 1}, of type {type(val)}, is not among {source_types}')
                else:
                    raise AssertionError(f'{candidate_desc} #{i + 1} is {val}, but only the following types are supported: {supported_types}')
        return matched_elements_positions
    matched_input_elements_positions = produce_matching(sources={'original inputs': flat_args}, candidates={'graph-captured input': graph_captured_input})
    flat_results_traced, out_spec_traced = pytree.tree_flatten(dynamo_traced_result)
    assert graph_captured_output is not None
    matched_output_elements_positions = produce_matching(sources={'graph-captured outputs': list(graph_captured_output), 'original inputs': flat_args}, candidates={'original output': flat_results_traced})
    new_graph = FlattenInputOutputSignature(graph, flat_args, matched_input_elements_positions, matched_output_elements_positions, example_fake_inputs, flat_args_dynamic_dims, fake_mode).transform()

    def argument_names(f_sig, args, kwargs) -> List[str]:

        def signature_to_fullargspec(sig: inspect.Signature):
            params = list(sig.parameters.values())
            args = [p.name for p in params if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
            kwonlyargs = [p.name for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY]
            varargs = next((p.name for p in params if p.kind == inspect.Parameter.VAR_POSITIONAL), None)
            varkw = next((p.name for p in params if p.kind == inspect.Parameter.VAR_KEYWORD), None)
            defaults = tuple((p.default for p in params if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and p.default is not inspect.Parameter.empty))
            kwonlydefaults = {p.name: p.default for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY and p.default is not inspect.Parameter.empty}
            annotations = {}
            if sig.return_annotation:
                annotations = {'return': sig.return_annotation}
            for parameter in params:
                annotations[parameter.name] = parameter.annotation
            return inspect.FullArgSpec(args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations)
        fullargspec = signature_to_fullargspec(f_sig)
        input_strs = fullargspec.args[:len(args)]
        if len(args) > len(fullargspec.args):
            assert fullargspec.varargs is not None, 'More arguments than expected'
            input_strs += [f'{fullargspec.varargs}_{i}' for i in range(0, len(args) - len(input_strs))]
        elif len(args) < len(fullargspec.args):
            for unprovided_arg in fullargspec.args[len(args):-len(fullargspec.defaults or [])]:
                assert unprovided_arg in kwargs, f'Missing argument {unprovided_arg}'
        input_strs += list(kwargs.keys())
        for kwonly_arg in fullargspec.kwonlyargs:
            kwonlydefaults = fullargspec.kwonlydefaults or {}
            assert kwonly_arg in kwargs or kwonly_arg in kwonlydefaults, f'Missing keyword only argument {kwonly_arg}'
        return input_strs
    new_graph.graph._codegen = _PyTreeCodeGen(_PyTreeInfo(argument_names(f_sig, orig_args, orig_kwargs), in_spec, out_spec_traced))
    new_graph.recompile()
    return new_graph