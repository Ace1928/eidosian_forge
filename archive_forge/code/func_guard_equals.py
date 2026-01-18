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
def guard_equals(self, left: Expr, right: Expr) -> Expr:
    if isinstance(left, Expr):
        left = sympy_subs(left, self.inv_precomputed_replacements)
    if isinstance(right, Expr):
        right = sympy_subs(right, self.inv_precomputed_replacements)
    assert self.shape_env.evaluate_expr(sympy.Eq(left, right))
    return left