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
def dim_can_vector(self, dim):
    """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Dimensional representation in terms of the canonical base dimensions.
        """
    vec = []
    for d in self.list_can_dims:
        vec.append(self.get_dimensional_dependencies(dim).get(d, 0))
    return Matrix(vec)