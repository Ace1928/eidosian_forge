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
@TorchPatcher.suppress_torch_distributed_warnings
def has_tensor_in_frame(frame):
    """Check if the frame has torch.* related bits"""
    if frame.f_code in always_optimize_code_objects:
        return True
    for co_name in frame.f_code.co_names:
        if co_name in frame.f_globals:
            obj = frame.f_globals[co_name]
            if is_allowed(obj):
                return True
            if np and config.trace_numpy and (obj is np or is_numpy(obj)):
                return True
    seen_ids: Dict[int, bool] = dict()

    def has_tensor(obj):
        """Recursively check if the obj has a tensor"""
        obj_id = id(obj)
        if obj_id in seen_ids:
            return seen_ids[obj_id]
        seen_ids[obj_id] = False
        if isinstance(obj, (torch.Tensor, torch.nn.Module)) or (istype(obj, type) and issubclass(obj, torch.nn.Module)):
            seen_ids[obj_id] = True
            return seen_ids[obj_id]
        elif config.trace_numpy and np and (istype(obj, np.ndarray) or isinstance(obj, np.generic)):
            seen_ids[obj_id] = True
            return seen_ids[obj_id]
        elif istype(obj, (list, tuple)):
            seen_ids[obj_id] = any((has_tensor(v) for v in obj))
            return seen_ids[obj_id]
        elif istype(obj, dict):
            values = list(obj.values())
            seen_ids[obj_id] = any((has_tensor(v) for v in values))
            return seen_ids[obj_id]
        elif istype(obj, (str, int, float, type(None), bool)):
            seen_ids[obj_id] = False
            return seen_ids[obj_id]
        elif is_namedtuple(obj):
            seen_ids[obj_id] = any((has_tensor(getattr(obj, v)) for v in obj._fields))
            return seen_ids[obj_id]
        else:
            return False
    for value in frame.f_locals.values():
        if has_tensor(value):
            return True
    log.debug('skipping because no torch.* %s             %s %s', frame.f_code.co_name, frame.f_code.co_filename, frame.f_code.co_firstlineno)
    return False