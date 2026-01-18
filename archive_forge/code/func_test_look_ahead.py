from sympy.combinatorics.fp_groups import FpGroup
from sympy.combinatorics.coset_table import (CosetTable,
from sympy.combinatorics.coset_table import modified_coset_enumeration_r
from sympy.combinatorics.free_groups import free_group
from sympy.testing.pytest import slow
def test_look_ahead():
    F, a, b, c = free_group('a, b, c')
    f = FpGroup(F, [a ** 11, b ** 5, c ** 4, (a * c) ** 3, b ** 2 * c ** (-1) * b ** (-1) * c, a ** 4 * b ** (-1) * a ** (-1) * b])
    H = [c, b, c ** 2]
    table0 = [[1, 2, 0, 0, 0, 0], [3, 0, 4, 5, 6, 7], [0, 8, 9, 10, 11, 12], [5, 1, 10, 13, 14, 15], [16, 5, 16, 1, 17, 18], [4, 3, 1, 8, 19, 20], [12, 21, 22, 23, 24, 1], [25, 26, 27, 28, 1, 24], [2, 10, 5, 16, 22, 28], [10, 13, 13, 2, 29, 30]]
    CosetTable.max_stack_size = 10
    C_c = coset_enumeration_c(f, H)
    C_c.compress()
    C_c.standardize()
    assert C_c.table[:10] == table0