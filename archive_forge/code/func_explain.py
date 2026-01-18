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
@patch('torch._dynamo.symbolic_convert.explain', True)
def explain(f, *extra_args, **extra_kwargs):

    def inner(*args, **kwargs):
        from . import reset
        reset()
        graphs: List[torch.fx.GraphModule] = []
        break_reasons: List[Any] = []
        op_count: int = 0
        ops_per_graph: List[torch.fx.Node] = []
        out_guards: List[_guards.Guard] = []

        def dynamo_graph_accumulating_compiler(gm: torch.fx.GraphModule, example_inputs):
            from .backends.debugging import _explain_graph_detail
            nonlocal graphs
            nonlocal op_count
            nonlocal ops_per_graph
            nonlocal break_reasons
            gm, graphs, op_count, ops_per_graph, break_reasons = _explain_graph_detail(gm, graphs, op_count, ops_per_graph, break_reasons)
            return gm.forward

        def guard_export_print(guards):
            nonlocal out_guards
            out_guards.extend(guards)
        opt_f = optimize(dynamo_graph_accumulating_compiler, nopython=False, guard_export_fn=guard_export_print)(f)
        opt_f(*args, **kwargs)
        graph_count = len(graphs)
        deduped_reasons = {}
        for reason in break_reasons:
            innermost_frame = reason.user_stack[-1]
            deduped_reasons[repr(innermost_frame)] = reason
        formatted_list = ''
        for idx, break_reason in enumerate(deduped_reasons.values()):
            formatted_stack = ''.join(traceback.format_list(break_reason.user_stack))
            msg = f'{idx + 1}. Reason: {break_reason.reason}\n   User Stack: {formatted_stack}\n'
            formatted_list += msg
        graph_break_count = graph_count - 1
        compile_time = compile_times(repr='str')
        reset()
        from .backends.debugging import ExplainOutput
        return ExplainOutput(graphs, graph_count, graph_break_count, break_reasons, op_count, ops_per_graph, out_guards, compile_time)
    if extra_args or extra_kwargs:
        warnings.warn("explain(f, *args, **kwargs) is deprecated, use explain(f)(*args, **kwargs) instead.  If you don't migrate, we may break your explain call in the future if your user defined kwargs conflict with future kwargs added to explain(f).")
        return inner(*extra_args, **extra_kwargs)
    else:
        return inner