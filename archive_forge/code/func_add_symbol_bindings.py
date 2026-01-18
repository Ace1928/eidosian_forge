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
def add_symbol_bindings(self, arg: GraphArg):
    if self.export:
        return
    assert arg.fake_tensor is not None

    def bind_symint(s, prop):
        if not (is_symbolic(s) and isinstance(s.node.expr, sympy.Symbol)):
            return
        s0 = s.node.expr
        if s0 in self.bound_symbols:
            return
        self.bound_symbols.add(s0)
        log.debug('bind_symint %s %s', s, prop.name())
        proxy = self.root_tracer.create_graph_input(str(s0), torch.SymInt, before=True, source=prop)
        proxy.node.meta['example_value'] = s
        proxy.node.meta['grapharg'] = GraphArg(prop, s, is_unspecialized=False, fake_tensor=None, is_tensor=False)

    def handle_tensor(t, src):
        for i, s in enumerate(t.size()):
            bind_symint(s, TensorPropertySource(src, TensorProperty.SIZE, i))
        for i, s in enumerate(t.stride()):
            bind_symint(s, TensorPropertySource(src, TensorProperty.STRIDE, i))
        bind_symint(t.storage_offset(), TensorPropertySource(src, TensorProperty.STORAGE_OFFSET))
        if is_traceable_wrapper_subclass(t):
            attrs, ctx = t.__tensor_flatten__()
            for attr in attrs:
                inner_t = getattr(t, attr)
                handle_tensor(inner_t, AttrSource(src, attr))
    handle_tensor(arg.fake_tensor, arg.source)