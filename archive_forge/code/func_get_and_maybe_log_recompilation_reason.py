from __future__ import annotations
import ast
import builtins
import collections
import dataclasses
import enum
import functools
import importlib
import inspect
import itertools
import logging
import math
import os
import re
import sys
import textwrap
import types
import weakref
from inspect import currentframe, getframeinfo
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from weakref import ReferenceType
import torch
import torch.utils._device
from torch._dynamo.source import (
from torch._guards import (
from torch.fx.experimental.symbolic_shapes import (
from torch.utils._traceback import format_frame, report_compile_source_on_error
from torch.utils.weak import TensorWeakRef
from . import config, convert_frame, exc, mutation_guard
from .eval_frame import set_guard_error_hook
from .source import DefaultsSource, LocalSource, TypeSource
from .types import GuardedCode, GuardFail, GuardFn  # noqa: F401
from .utils import (
def get_and_maybe_log_recompilation_reason(cache_entry, frame: types.FrameType) -> List[str]:
    """
    Return the list of guard failure reasons using cache_entry.
    Logs the recompilation reason if `recompiles` logging is enabled.
    Raises a RecompileError if `config.error_on_recompile` is enabled.
    """
    reasons = []
    while cache_entry is not None:
        reason = get_guard_fail_reason(cache_entry.check_fn, cache_entry.code, frame.f_locals)
        if reason:
            reasons.append(reason)
        cache_entry = cache_entry.next
    code = frame.f_code
    do_recompiles_log = is_recompiles_enabled() or is_recompiles_verbose_enabled()
    if do_recompiles_log or config.error_on_recompile:
        if is_recompiles_verbose_enabled():
            failures = '\n\n'.join((f'guard {i} failures:\n' + textwrap.indent(reason, '- ') for i, reason in enumerate(reasons)))
        else:
            failures = textwrap.indent('\n'.join(reasons), '- ')
        guard_failure_details = f'triggered by the following guard failure(s):\n{failures}'
        message = f'Recompiling function {code.co_name} in {code.co_filename}:{code.co_firstlineno}\n{textwrap.indent(guard_failure_details, '    ')}'
        if do_recompiles_log:
            if is_recompiles_verbose_enabled():
                recompiles_verbose_log.debug(message)
            else:
                recompiles_log.debug(message)
        if config.error_on_recompile:
            raise exc.RecompileError(message)
    return reasons