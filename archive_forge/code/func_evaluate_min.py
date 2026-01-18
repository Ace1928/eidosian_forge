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
def evaluate_min(self, left: Expr, right: Expr) -> Expr:
    """return the smaller of left and right, and guard on that choice"""
    lv = self.size_hint(left)
    rv = self.size_hint(right)
    if lv <= rv:
        self.guard_leq(left, right)
        return left
    else:
        self.guard_leq(right, left)
        return right