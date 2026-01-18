import collections
import contextlib
import copy
import functools
import itertools
import logging
import operator
import re
import sys
import traceback
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union
import sympy
import torch._guards
import torch._logging
import torch.nn
import torch.utils._pytree as pytree
from torch import fx
from torch._guards import (
from torch._utils_internal import signpost_event
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import free_symbols, is_symbolic, ShapeEnv
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.interp import sympy_interp
from torch.utils._sympy.reference import PythonReferenceAnalysis
from torch.utils.weak import WeakTensorKeyDictionary
from . import config, logging as torchdynamo_logging, variables
from .backends.registry import CompiledFn, CompilerFn
from .bytecode_transformation import (
from .code_context import code_context
from .codegen import PyCodegen
from .current_scope_id import enter_new_scope
from .exc import (
from .guards import GuardBuilder, install_guard
from .mutation_guard import is_dynamic_nn_module
from .side_effects import SideEffects
from .source import (
from .utils import (
from .variables.base import VariableTracker
from .variables.builder import GraphArg, TrackedFake, VariableBuilder, wrap_fx_proxy
from .variables.nn_module import NNModuleVariable
from .variables.tensor import (
from .variables.torch_function import TensorWithTFOverrideVariable
def register_attr_or_module(self, target: Union[torch.nn.Module, torch.Tensor, Any], *names, **options):
    if is_dynamic_nn_module(target):
        return variables.UnspecializedNNModuleVariable(target, **options)
    options = dict(options)
    assert 'source' in options
    source = options['source']
    assert not isinstance(source, ParamBufferSource)
    if isinstance(target, torch.Tensor):
        tracer = self.current_tracer
        if not self.is_root_tracer():
            tracer = self.root_tracer
        if not is_constant_source(source):
            install_guard(source.make_guard(GuardBuilder.TENSOR_MATCH))
        if get_static_address_type(target) == 'guarded':
            install_guard(source.make_guard(GuardBuilder.DATA_PTR_MATCH))

        def wrap_name(module_key):
            assert self.param_name_to_source is not None
            self.param_name_to_source[module_key] = source
            return wrap_fx_proxy(self.root_tx, tracer.create_proxy('get_attr', module_key, tuple(), {}), example_value=target, **options)
    elif isinstance(target, torch.nn.Module):
        assert isinstance(target, torch.nn.Module)
        install_guard(source.make_guard(GuardBuilder.NN_MODULE))

        def wrap_name(module_key):
            return NNModuleVariable(type(target), module_key, **options)
    elif isinstance(target, (torch.SymInt, torch.SymFloat)):

        def wrap_name(module_key):
            return SymNodeVariable.create(self, self.create_proxy('get_attr', module_key, tuple(), {}), sym_num=target, **options)
    else:

        def wrap_name(module_key):
            self.output.update_co_names(module_key)
            self.global_scope[module_key] = target
            return VariableBuilder(self, ConstantSource(source_name=module_key))(target)
    for k, v in self.nn_modules.items():
        if v is target:
            return wrap_name(k)
    name = OutputGraph.module_key_name(*names)
    base = name
    for i in itertools.count():
        if name not in self.nn_modules:
            self.nn_modules[name] = target
            if isinstance(target, torch.nn.Module):

                def register_leaf_name(leaf_name):
                    assert self.param_name_to_source is not None
                    new_source = ParamBufferSource(source, leaf_name)
                    new_name = f'{name}.{leaf_name}'
                    self.param_name_to_source[new_name] = new_source
                if hasattr(target, '_parameters'):
                    for leaf_name, _ in target.named_parameters():
                        register_leaf_name(leaf_name)
                if hasattr(target, '_buffers'):
                    for leaf_name, _ in target.named_buffers():
                        register_leaf_name(leaf_name)
            return wrap_name(name)
        name = f'{base}_{i}'
    raise AssertionError('unreachable')