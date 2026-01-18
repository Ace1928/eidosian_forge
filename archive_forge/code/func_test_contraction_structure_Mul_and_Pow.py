from sympy.core import symbols, S, Pow, Function
from sympy.functions import exp
from sympy.testing.pytest import raises
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.tensor.index_methods import IndexConformanceException
from sympy.tensor.index_methods import (get_contraction_structure, get_indices)
def test_contraction_structure_Mul_and_Pow():
    x = IndexedBase('x')
    y = IndexedBase('y')
    i, j, k = (Idx('i'), Idx('j'), Idx('k'))
    i_ji = x[i] ** (y[j] * x[i])
    assert get_contraction_structure(i_ji) == {None: {i_ji}}
    ij_i = (x[i] * y[j]) ** y[i]
    assert get_contraction_structure(ij_i) == {None: {ij_i}}
    j_ij_i = x[j] * (x[i] * y[j]) ** y[i]
    assert get_contraction_structure(j_ij_i) == {(j,): {j_ij_i}}
    j_i_ji = x[j] * x[i] ** (y[j] * x[i])
    assert get_contraction_structure(j_i_ji) == {(j,): {j_i_ji}}
    ij_exp_kki = x[i] * y[j] * exp(y[i] * y[k, k])
    result = get_contraction_structure(ij_exp_kki)
    expected = {(i,): {ij_exp_kki}, ij_exp_kki: [{None: {exp(y[i] * y[k, k])}, exp(y[i] * y[k, k]): [{None: {y[i] * y[k, k]}, y[i] * y[k, k]: [{(k,): {y[k, k]}}]}]}]}
    assert result == expected