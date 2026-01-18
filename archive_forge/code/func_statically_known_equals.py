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
def statically_known_equals(self, left: Expr, right: Expr) -> bool:
    """
        Returns a bool indicating if it is sound to optimize as if left and right are equal.
        """
    return self.is_expr_static_and_true(sympy.Eq(left, right))