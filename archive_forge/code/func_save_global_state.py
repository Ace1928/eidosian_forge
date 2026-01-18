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
def save_global_state(self, out=None):
    """
        Saves to out if it is provided. Else saves to the tracing context's global_state.
        """
    global_state = out if out is not None else self.tracing_context.global_context.global_state
    global_state['torch_function_enabled'] = (self.set_torch_function_state, self.torch_function_enabled)
    global_state['grad_enabled'] = (torch.set_grad_enabled, torch.is_grad_enabled())
    global_state['autocast_enabled'] = (torch.set_autocast_enabled, torch.is_autocast_enabled())
    global_state['autocast_cpu_enabled'] = (torch.set_autocast_cpu_enabled, torch.is_autocast_cpu_enabled())
    global_state['autocast_gpu_dtype'] = (torch.set_autocast_gpu_dtype, torch.get_autocast_gpu_dtype())
    global_state['autocast_cpu_dtype'] = (torch.set_autocast_cpu_dtype, torch.get_autocast_cpu_dtype())
    global_state['autocast_cache_enabled'] = (torch.set_autocast_cache_enabled, torch.is_autocast_cache_enabled())