from __future__ import annotations
import collections
from functools import reduce
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.core.expr import Expr
from sympy.core.power import Pow
def parse_dim(dim):
    if isinstance(dim, str):
        dim = Dimension(Symbol(dim))
    elif isinstance(dim, Dimension):
        pass
    elif isinstance(dim, Symbol):
        dim = Dimension(dim)
    else:
        raise TypeError('%s wrong type' % dim)
    return dim