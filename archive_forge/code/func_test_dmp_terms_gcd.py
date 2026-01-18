from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dmp_terms_gcd():
    assert dmp_terms_gcd([[]], 1, ZZ) == ((0, 0), [[]])
    assert dmp_terms_gcd([1, 0, 1, 0], 0, ZZ) == ((1,), [1, 0, 1])
    assert dmp_terms_gcd([[1], [], [1], []], 1, ZZ) == ((1, 0), [[1], [], [1]])
    assert dmp_terms_gcd([[1, 0], [], [1]], 1, ZZ) == ((0, 0), [[1, 0], [], [1]])
    assert dmp_terms_gcd([[1, 0], [1, 0, 0], [], []], 1, ZZ) == ((2, 1), [[1], [1, 0]])