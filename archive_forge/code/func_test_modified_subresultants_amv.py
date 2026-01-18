from sympy.core.symbol import var
from sympy.polys.polytools import (pquo, prem, sturm, subresultants)
from sympy.matrices import Matrix
from sympy.polys.subresultants_qq_zz import (sylvester, res, res_q, res_z, bezout,
def test_modified_subresultants_amv():
    x = var('x')
    p = x ** 8 + x ** 6 - 3 * x ** 4 - 3 * x ** 3 + 8 * x ** 2 + 2 * x - 5
    q = 3 * x ** 6 + 5 * x ** 4 - 4 * x ** 2 - 9 * x + 21
    amv_factors = [1, 1, -1, 1, -1, 1]
    assert modified_subresultants_amv(p, q, x) == [i * j for i, j in zip(amv_factors, subresultants_amv(p, q, x))]
    assert modified_subresultants_amv(p, q, x)[-1] != sylvester(p + x ** 8, q, x).det()
    assert modified_subresultants_amv(p, q, x) != sturm_amv(p, q, x)
    p = x ** 3 - 7 * x + 7
    q = 3 * x ** 2 - 7
    assert modified_subresultants_amv(p, q, x) == sturm_amv(p, q, x)
    assert modified_subresultants_amv(-p, q, x) != sturm_amv(-p, q, x)