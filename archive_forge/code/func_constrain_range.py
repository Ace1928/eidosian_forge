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
@record_shapeenv_event()
def constrain_range(a, *, min: Optional[int], max: Optional[int]=None):
    """
    Applies a constraint that the passed in SymInt must lie between min-max
    inclusive-inclusive, WITHOUT introducing a guard on the SymInt (meaning
    that it can be used on unbacked SymInts).  If min/max are None, we assume
    that the dimension is unbounded in that direction.  Repeated application
    of constrain_range intersects the ranges.  This is a fairly low level API
    that doesn't have a lot of safety guarantees (TODO: provide higher level
    APIs).

    Currently, we use this API in the following circumstance: when we allocate
    an unbacked SymInt, denoting an integer quantity which is data dependent,
    we ordinarily do not know anything about what values it may take.  This
    means that any sort of guard on it will immediately fail.  However, in
    many cases, we know something about the unbacked SymInt: for example, we
    know that nonzero(x).size(0) must be >= 0.  We use constrain_range to
    narrow the possible range, declaring that negative symbols are impossible.
    This permits to definitely answer True to queries like 'nnz >= 0', even if
    we don't know what the actual (hinted) value of 'nnz' is.  In fact, we
    actually use constrain_range to unsoundly discharge common guards: for an
    unbacked SymInt produced by nonzero, we will also assume that it is not
    equal to 0/1 (even though these are perfectly possible values at runtime),
    because we generally expect graphs that are valid for N=2 to also be valid
    for N=1.

    .. warning::
        If you use constrain_range in the context of tracing, we do NOT check
        that the constraint was actually valid at runtime!  In fact, we
        cannot (easily) do so, as we currently unsoundly assume that unbacked
        SymInt can never be zero/one, even if it may actually take on these
        values at runtime (we assume that a graph that is valid for N=2 will
        also be valid for N=1).
    """
    if min is None:
        min = -sympy.oo
    if max is None:
        max = sympy.oo
    if max < min:
        raise ValueError("Maximum value to constrain_as_size can't be less than the specified min value, received min={min} and max={max}")
    if isinstance(a, int):
        if not min <= a <= max:
            raise ValueError(f'Invalid value {a} for range [{min}:{max}]')
        return
    if isinstance(a.node.expr, sympy.Integer):
        if not min <= int(a.node.expr) <= max:
            raise ValueRangeError(f'Invalid value {int(a.node.expr)} for range [{min}:{max}]')
        return
    assert isinstance(a.node.expr, sympy.Symbol), 'constraining non-Symbols NYI'
    _constrain_symbol_range(a.node.shape_env, a.node.expr, compiler_min=min, compiler_max=max, runtime_min=min, runtime_max=max)