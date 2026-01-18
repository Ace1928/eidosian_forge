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
def lookup_precomputed_size(self, expr: Expr) -> sympy.Symbol:
    if expr not in self.precomputed_replacements:
        sym = sympy_symbol(f'ps{len(self.precomputed_replacements)}')
        self.precomputed_replacements[expr] = sym
        self.inv_precomputed_replacements[sym] = expr
    return self.precomputed_replacements[expr]