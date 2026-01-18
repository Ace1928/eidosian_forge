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
def is_dimensionless(self, dimension):
    """
        Check if the dimension object really has a dimension.

        A dimension should have at least one component with non-zero power.
        """
    if dimension.name == 1:
        return True
    return self.get_dimensional_dependencies(dimension) == {}