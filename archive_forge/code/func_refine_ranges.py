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
def refine_ranges(self, guard: ShapeGuard) -> None:
    expr = self.simplify(guard.expr)
    for symbol in expr.free_symbols:
        assert isinstance(symbol, sympy.Symbol)
        if isinstance(self.var_to_val.get(symbol, None), SingletonInt):
            continue
        r = try_solve(expr, symbol)
        if r is None or not (symbol.is_integer and r[1].is_integer):
            continue
        r_expr, rhs = r
        vr = self.var_to_range[symbol]
        lower, upper = (vr.lower, vr.upper)
        rhs_vr = bound_sympy(rhs, self.var_to_range)
        _assert_bound_is_rational(rhs, rhs_vr)
        lower_guard, upper_guard = self.var_to_guards.get(symbol, (None, None))
        if lower < rhs_vr.lower and isinstance(r_expr, (sympy.Eq, sympy.Ge, sympy.Gt)):
            lower = rhs_vr.lower + int(isinstance(r_expr, sympy.Gt))
            lower_guard = guard
        if upper > rhs_vr.upper and isinstance(r_expr, (sympy.Eq, sympy.Le, sympy.Lt)):
            upper = rhs_vr.upper - int(isinstance(r_expr, sympy.Lt))
            upper_guard = guard
        if vr == ValueRanges(lower, upper):
            continue
        self.var_to_range[symbol] = ValueRanges(lower, upper)
        self.var_to_guards[symbol] = (lower_guard, upper_guard)
        self._maybe_evaluate_static.cache_clear()