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
def size_hints(self, exprs: Iterable[Expr], *, fallback: Optional[int]=None) -> Tuple[int, ...]:
    return tuple((self.size_hint(x, fallback=fallback) for x in exprs))