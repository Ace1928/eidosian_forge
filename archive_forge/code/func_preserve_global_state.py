import collections
import functools
import itertools
import logging
import os
import random
import types
import typing
import weakref
from typing import Any, Callable, Dict, List, Optional, Set
import torch
import torch._logging
from torch._guards import compile_context, CompileContext, CompileId, tracing
from torch._utils_internal import log_compilation_event, signpost_event
from torch.fx.experimental.symbolic_shapes import (
from torch.fx.graph_module import _forward_from_src as original_forward_from_src
from torch.utils._traceback import format_traceback_short
from . import config, exc
from .allowed_functions import is_allowed, is_numpy
from .backends.registry import CompilerFn
from .bytecode_analysis import remove_dead_code, remove_pointless_jumps
from .bytecode_transformation import (
from .cache_size import (
from .eval_frame import always_optimize_code_objects, skip_code, TorchPatcher
from .exc import (
from .guards import (
from .hooks import Hooks
from .output_graph import OutputGraph
from .replay_record import ExecutionRecord
from .symbolic_convert import InstructionTranslator, SpeculationLog
from .types import BytecodeHook
from .utils import (
from collections import OrderedDict
from torch.utils.hooks import RemovableHandle
def preserve_global_state(fn):
    """
    Context manager to:
        1) Save/restore torch.is_grad_enabled() state
        2) Save/restore python random state
        3) Save/restore torch random state
        4) Monkey patch torch.fx.graph_module._forward_from_src
    """

    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        guards = GlobalStateGuard()
        prior_grad_mode = torch.is_grad_enabled()
        prior_inference_mode = torch.is_inference_mode_enabled()
        prior_deterministic = torch.are_deterministic_algorithms_enabled()
        prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        py_rng_state = random.getstate()
        torch_rng_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
        prior_fwd_from_src = torch.fx.graph_module._forward_from_src
        torch.fx.graph_module._forward_from_src = fx_forward_from_src_skip_result
        cleanup = setup_compile_debug()
        try:
            return fn(*args, **kwargs)
        finally:
            cleanup.close()
            torch._C._set_grad_enabled(prior_grad_mode)
            torch.torch.autograd.grad_mode._enter_inference_mode(prior_inference_mode)
            torch.use_deterministic_algorithms(prior_deterministic, warn_only=prior_warn_only)
            random.setstate(py_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)
            torch.fx.graph_module._forward_from_src = prior_fwd_from_src
            assert guards.check(), 'Global state changed while dynamo tracing, please report a bug'
    _fn._torchdynamo_orig_callable = fn
    return _fn