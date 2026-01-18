from __future__ import annotations
from functools import singledispatch
from math import prod
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.tensor.indexed import Indexed
from sympy.utilities.lambdify import lambdify
from sympy.core.relational import Relational
from sympy.core.sympify import _sympify
from sympy.sets.sets import FiniteSet, ProductSet, Intersection
from sympy.solvers.solveset import solveset
from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable
def rs_swap(a, b):
    """
    Build a dictionary to swap RandomSymbols based on their underlying symbol.

    i.e.
    if    ``X = ('x', pspace1)``
    and   ``Y = ('x', pspace2)``
    then ``X`` and ``Y`` match and the key, value pair
    ``{X:Y}`` will appear in the result

    Inputs: collections a and b of random variables which share common symbols
    Output: dict mapping RVs in a to RVs in b
    """
    d = {}
    for rsa in a:
        d[rsa] = [rsb for rsb in b if rsa.symbol == rsb.symbol][0]
    return d