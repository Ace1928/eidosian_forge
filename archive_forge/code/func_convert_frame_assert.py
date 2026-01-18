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
def convert_frame_assert(compiler_fn: CompilerFn, one_graph: bool=True, export: bool=False, export_constraints=None):
    """Fully convert a frame into an FX graph"""
    reset_graph_break_dup_checker()

    def _convert_frame_assert(frame: types.FrameType, cache_entry, hooks: Hooks, frame_state):
        increment_frame()
        code = frame.f_code
        cache_size = compute_cache_size(frame, cache_entry)
        recompile_reasons = None
        if is_recompilation(cache_size):
            recompile_reasons = get_and_maybe_log_recompilation_reason(cache_entry, frame)
        input_codes.add(code)
        if code in output_codes:
            return None
        if os.environ.get('TORCHDYNAMO_DEBUG_FUNCTION') and os.environ.get('TORCHDYNAMO_DEBUG_FUNCTION') != code.co_name:
            return None
        if code.co_name == '<genexpr>' and code.co_filename.endswith(('transformers/file_utils.py', 'transformers/utils/generic.py', 'diffusers/utils/outputs.py')):
            return None
        if code.co_name == '__setattr__':
            return None
        if code.co_name == '__init__' and code.co_filename.startswith(os.path.dirname(torch.optim.__file__)):
            return None
        if code.co_name == '<module>' and code.co_filename == '<string>':
            return None
        if code.co_name == '<lambda>' and code.co_filename == '<string>' and (not bool(frame.f_builtins)):
            return None
        if is_generator(code):
            unimplemented('generator')
        if exceeds_cache_size_limit(cache_size):

            def format_func_info(code):
                return f"'{code.co_name}' ({code.co_filename}:{code.co_firstlineno})"

            def format_guard_failures():
                assert recompile_reasons, 'TODO(whc) any other recompile reasons?'
                return recompile_reasons[-1]
            log.warning('torch._dynamo hit config.cache_size_limit (%s)\n   function: %s\n   last reason: %s\nTo log all recompilation reasons, use TORCH_LOGS="recompiles".\nTo diagnose recompilation issues, see %s.', config.cache_size_limit, format_func_info(code), format_guard_failures(), troubleshooting_url)
            unimplemented('cache_size_limit reached')
        if not has_tensor_in_frame(frame):
            return None
        global initial_global_state
        initial_global_state = GlobalStateGuard()
        global FRAME_COUNTER
        if '_id' not in frame_state:
            frame_state['_id'] = FRAME_COUNTER
            FRAME_COUNTER += 1
        frame_id = frame_state['_id']
        frame_compile_id = FRAME_COMPILE_COUNTER[frame_id]
        FRAME_COMPILE_COUNTER[frame_id] += 1
        compile_id = CompileId(frame_id, frame_compile_id)
        signpost_event('dynamo', '_convert_frame_assert._compile', {'co_name': code.co_name, 'co_filename': code.co_filename, 'co_firstlineno': code.co_firstlineno, 'cache_size': cache_size.num_cache_entries_in_bucket, 'accumulated_cache_size': cache_size.num_cache_entries})
        with config.patch(_patch_config_if_changed()):
            compiled_product = _compile(frame.f_code, frame.f_globals, frame.f_locals, frame.f_builtins, compiler_fn, one_graph, export, export_constraints, hooks, cache_size, frame, frame_state=frame_state, compile_id=compile_id)
        return compiled_product
    _convert_frame_assert._torchdynamo_orig_callable = compiler_fn

    def _clone_with_backend(backend):
        return convert_frame_assert(backend, one_graph, export, export_constraints)
    _convert_frame_assert._clone_with_backend = _clone_with_backend
    return _convert_frame_assert