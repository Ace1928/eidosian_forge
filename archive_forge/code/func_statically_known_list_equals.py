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
def statically_known_list_equals(self, left: List[Expr], right: List[Expr]) -> bool:
    """
        Returns a bool indicating if it is sound to optimize as if left and right lists are equal.
        """
    if len(left) != len(right):
        return False
    if all((self.statically_known_equals(l, r) for l, r in zip(left, right))):
        return True
    return False