from sympy.core import symbols, S, Pow, Function
from sympy.functions import exp
from sympy.testing.pytest import raises
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.tensor.index_methods import IndexConformanceException
from sympy.tensor.index_methods import (get_contraction_structure, get_indices)
def test_contraction_structure_Add_in_Pow():
    x = IndexedBase('x')
    y = IndexedBase('y')
    i, j, k = (Idx('i'), Idx('j'), Idx('k'))
    s_ii_jj_s = (1 + x[i, i]) ** (1 + y[j, j])
    expected = {None: {s_ii_jj_s}, s_ii_jj_s: [{None: {S.One}, (i,): {x[i, i]}}, {None: {S.One}, (j,): {y[j, j]}}]}
    result = get_contraction_structure(s_ii_jj_s)
    assert result == expected
    s_ii_jk_s = (1 + x[i, i]) ** (1 + y[j, k])
    expected_2 = {None: {(x[i, i] + 1) ** (y[j, k] + 1)}, s_ii_jk_s: [{None: {S.One}, (i,): {x[i, i]}}]}
    result_2 = get_contraction_structure(s_ii_jk_s)
    assert result_2 == expected_2