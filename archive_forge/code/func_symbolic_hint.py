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
def symbolic_hint(self, expr: Expr) -> Expr:
    if not isinstance(expr, Expr):
        assert isinstance(expr, int)
        return expr
    free_symbols = expr.free_symbols
    if not free_symbols:
        return int(expr)
    while any((s.name.startswith('ps') for s in free_symbols)):
        expr = sympy_subs(expr, self.inv_precomputed_replacements)
        free_symbols = expr.free_symbols
    return sympy_subs(expr, self.var_to_val)