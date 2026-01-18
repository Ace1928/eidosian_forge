import functools
import itertools
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import sympy
from sympy import Expr
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import bound_sympy
from .utils import sympy_subs, sympy_symbol, VarRanges
from .virtualized import V
def remove_zero_terms(base, divisor):
    """Symbols smaller than the divisor are zero"""
    for v in base.free_symbols:
        if v in var_ranges:
            rest = sympy.Wild('_rest', exclude=[v])
            m = base.match(v + rest)
            if m and v not in m[rest].free_symbols:
                gcd = sympy.gcd(m[rest], divisor)
                if gcd == divisor:
                    if self.statically_known_leq(var_ranges[v], divisor):
                        base = m[rest]
    return base