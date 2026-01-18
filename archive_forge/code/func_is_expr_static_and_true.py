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
def is_expr_static_and_true(self, expr: Union[Expr, int]) -> bool:
    if expr in (True, False):
        return bool(expr)
    try:
        simplified = self.shape_env._maybe_evaluate_static(expr)
        if simplified is not None:
            return bool(simplified)
    except Exception:
        log.debug('Could not simplify %s', expr)
    return False