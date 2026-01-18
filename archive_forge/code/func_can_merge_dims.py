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
def can_merge_dims(a, b):
    for k in range(len(strides)):
        if self.simplify(strides[k][a] * sizes[a]) == self.simplify(strides[k][b]):
            va = index_vars[a]
            vb = index_vars[b]
            v = sympy_symbol('_merge_tester')
            expr1 = sympy_subs(index_formulas[k], {va: v * sizes[a], vb: 0})
            expr2 = sympy_subs(index_formulas[k], {va: 0, vb: v})
            if self.simplify(expr1) == self.simplify(expr2):
                continue
        return False
    return True