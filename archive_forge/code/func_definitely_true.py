import builtins
import collections
import functools
import inspect
import itertools
import logging
import math
import operator
import re
import sys
import threading
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, cast, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, Iterable
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.fx.experimental import _config as config
from torch.fx.experimental.recording import (
from torch.fx.experimental.sym_node import SymNode, SymTypes
from torch import SymBool, SymFloat, SymInt
from torch._guards import ShapeGuard, Source, TracingContext
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.functions import FloorDiv, Mod, IsNonOverlappingAndDenseIndicator
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.value_ranges import bound_sympy, SymPyValueRangeAnalysis, ValueRanges, ValueRangeError
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._traceback import format_frame, CapturedTraceback
from torch._utils_internal import signpost_event
from torch._logging import LazyString
import sympy
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE
def definitely_true(a):
    """
    Returns True only if we can tell that a is True, possibly introducing
    a guard in the process.  If a depends on some unbacked SymInt, we may
    return False even though there may exist a possible value of the SymInt
    that would cause the expression to return True.

    When is it appropriate to use definitely_true?  First, if you can use
    a higher level combinator like parallel_or/parallel_and, prefer using
    those instead, they are definitely safe (modulo short-circuiting).
    Second, it can be used if the program would behave equivalently if
    definitely_true always returned False (parallel_or/parallel_and are
    examples of this pattern, modulo short-circuiting).  Finally, it even
    be OK if the program wouldn't behave equivalently, so long as the
    change is semantics preserving.  It can be semantics preserving if
    the program errors in more cases than it did previously (but otherwise
    behaves identically), or if it changes some quantity in a way that
    doesn't matter (e.g., strides often fall in this bucket.)
    """
    if isinstance(a, SymBool):
        if a.node.has_hint():
            return guard_bool(a)
        else:
            return False
    return bool(a)