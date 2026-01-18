import dataclasses
import itertools
import sympy
from sympy.logic.boolalg import BooleanAtom, Boolean as SympyBoolean
import operator
import math
import logging
import torch
from typing import Union, Dict, Optional, SupportsFloat
from torch._prims_common import dtype_to_type
from .interp import sympy_interp
def bound_sympy(expr: sympy.Expr, ranges: Optional[Dict[sympy.Symbol, ValueRanges]]=None) -> ValueRanges:
    if isinstance(expr, sympy.Number):
        return ValueRanges.wrap(expr)
    ranges = ranges or {}
    context = torch._guards.TracingContext.try_get()
    if context and context.fake_mode.shape_env:
        ranges = {**ranges, **context.fake_mode.shape_env.var_to_range}
    unbounded_vars = expr.free_symbols - ranges.keys()
    if unbounded_vars:
        unbounded_ranges: Dict[sympy.Symbol, ValueRanges] = {}
        for s in unbounded_vars:
            assert s.is_integer
            if s.is_positive:
                lower = 1
            elif s.is_nonnegative:
                lower = 0
            else:
                lower = -math.inf
            unbounded_ranges[s] = ValueRanges(lower, math.inf)
        ranges = {**ranges, **unbounded_ranges}
    return sympy_interp(SymPyValueRangeAnalysis, ranges, expr)