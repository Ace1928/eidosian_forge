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
def arg_ref(self, guard: Union[str, Guard]) -> str:
    name: str
    if isinstance(guard, str):
        name = guard
    else:
        name = guard.name
    base = strip_getattr_getitem(strip_function_call(name))
    if base not in self.argnames:
        if re.match('[a-zA-Z0-9_]+', base):
            if re.match('^\\d+$', base):
                log.warning('invalid var name: %s', guard)
            self.argnames.append(base)
    return name