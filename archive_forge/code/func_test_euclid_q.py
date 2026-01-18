from sympy.core.symbol import var
from sympy.polys.polytools import (pquo, prem, sturm, subresultants)
from sympy.matrices import Matrix
from sympy.polys.subresultants_qq_zz import (sylvester, res, res_q, res_z, bezout,
def test_euclid_q():
    x = var('x')
    p = x ** 3 - 7 * x + 7
    q = 3 * x ** 2 - 7
    assert euclid_q(p, q, x)[-1] == -sturm(p)[-1]