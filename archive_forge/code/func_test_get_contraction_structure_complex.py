from sympy.core import symbols, S, Pow, Function
from sympy.functions import exp
from sympy.testing.pytest import raises
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.tensor.index_methods import IndexConformanceException
from sympy.tensor.index_methods import (get_contraction_structure, get_indices)
def test_get_contraction_structure_complex():
    x = IndexedBase('x')
    y = IndexedBase('y')
    A = IndexedBase('A')
    i, j, k = (Idx('i'), Idx('j'), Idx('k'))
    expr1 = y[i] + A[i, j] * x[j]
    d1 = {None: {y[i]}, (j,): {A[i, j] * x[j]}}
    assert get_contraction_structure(expr1) == d1
    expr2 = expr1 * A[k, i] + x[k]
    d2 = {None: {x[k]}, (i,): {expr1 * A[k, i]}, expr1 * A[k, i]: [d1]}
    assert get_contraction_structure(expr2) == d2