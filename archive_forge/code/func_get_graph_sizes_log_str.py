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
def get_graph_sizes_log_str(self, name):
    graph_sizes_str = 'TRACED GRAPH TENSOR SIZES\n'
    graph_sizes_str += f'===== {name} =====\n'
    for node in self.graph.nodes:
        example_value = node.meta.get('example_value', None)
        if isinstance(example_value, torch._subclasses.FakeTensor):
            size = example_value.size()
            graph_sizes_str += f'{node.name}: {tuple(size)}\n'
            concrete_size = []
            has_symint = False
            for sz in size:
                if isinstance(sz, int):
                    concrete_size.append(sz)
                elif isinstance(sz, torch.SymInt):
                    has_symint = True
                    concrete_size.append(sz.node.hint)
                else:
                    break
            else:
                if has_symint:
                    graph_sizes_str += f'{node.name} (concrete): {tuple(concrete_size)}\n'
    return graph_sizes_str