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
@staticmethod
@functools.lru_cache(None)
def patch():
    from .decorators import disable
    torch.jit.trace = disable(torch.jit.trace)
    torch.jit.trace_module = disable(torch.jit.trace_module)
    torch.jit._get_trace_graph = disable(torch.jit._get_trace_graph)
    torch.fx._symbolic_trace.Tracer.trace = disable(torch.fx._symbolic_trace.Tracer.trace)
    torch.distributions.Distribution.set_default_validate_args(False)
    from ..optim import adadelta, adagrad, adam, adamax, adamw, asgd, lbfgs, nadam, radam, rmsprop, rprop, sgd, sparse_adam
    optimizer_modules = {adadelta, adagrad, adam, adamax, adamw, asgd, lbfgs, nadam, radam, rmsprop, rprop, sgd, sparse_adam}
    disabled_multi_tensor_opt_modules = {adamax, radam, sgd}
    for opt_mod in optimizer_modules:
        opt_name = opt_mod.__name__.split('.')[-1]
        multi_tensor_fn_name = f'_multi_tensor_{opt_name}'
        fused_fn_name = f'_fused_{opt_name}'
        if hasattr(opt_mod, multi_tensor_fn_name) and opt_mod in disabled_multi_tensor_opt_modules:
            setattr(opt_mod, multi_tensor_fn_name, disable(getattr(opt_mod, multi_tensor_fn_name)))
        if hasattr(opt_mod, fused_fn_name):
            setattr(opt_mod, fused_fn_name, disable(getattr(opt_mod, fused_fn_name)))
    optimizer_classes = [opt for opt in torch.optim.__dict__.values() if inspect.isclass(opt) and issubclass(opt, torch.optim.Optimizer)]
    excluded_optimizer_classes = {torch.optim.SparseAdam, torch.optim.RAdam, torch.optim.LBFGS}
    for opt in optimizer_classes:
        if opt in excluded_optimizer_classes:
            opt.step = disable(opt.step)
        if hasattr(opt, '_init_group'):
            opt._init_group = disable(opt._init_group)
        hooked = getattr(opt.step, 'hooked', False)
        if hooked:
            unwrapped_step = getattr(opt.step, '__wrapped__', None)
            if unwrapped_step:
                opt.step = unwrapped_step
        opt.step.hooked = True