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
def stride_hints(self, index: Expr, vars: List[sympy.Symbol], support_vars: Optional[List[sympy.Symbol]]=None) -> List[int]:
    for v in index.free_symbols:
        if v.name.startswith('indirect'):
            index = sympy_subs(index, {v: 0})
    result = []
    for s in self.stride_vars(index, vars, support_vars):
        try:
            result.append(self.size_hint(s))
        except TypeError:
            result.append(0)
    return result