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
def get_guard_fail_reason(guard_fn: GuardFn, code: types.CodeType, f_locals: Dict[str, object]) -> str:
    """
    Return the reason why `guard_fn` failed.
    Updates `guard_failures` with the generated reason.
    Only the first failed check of guard_fn is reported.
    """
    scope = {'L': f_locals, 'G': guard_fn.global_scope['G']}
    scope.update(guard_fn.closure_vars)
    scope['___check_tensors'] = scope['___check_tensors_verbose']
    reasons: List[str] = []
    for part in guard_fn.verbose_code_parts:
        global_scope = dict(guard_fn.global_scope)
        global_scope['__compile_source__'] = part
        with report_compile_source_on_error():
            try:
                fail_reason = eval(part, global_scope, scope)
            except Exception as e:
                if is_recompiles_verbose_enabled():
                    continue
                else:
                    raise
        if isinstance(fail_reason, bool) and (not fail_reason):
            fail_reason = part
        if isinstance(fail_reason, str):
            reasons.append(fail_reason)
            if not is_recompiles_verbose_enabled():
                break
    reason_str = '\n'.join(reasons)
    guard_failures[orig_code_map[code]].append(reason_str)
    try:
        if guard_fn.guard_fail_fn is not None:
            guard_fn.guard_fail_fn(GuardFail(reason_str or 'unknown reason', orig_code_map[code]))
    except Exception as e:
        log.exception('Failure in guard_fail_fn callback - raising here will cause a NULL Error on guard eval')
    return reason_str