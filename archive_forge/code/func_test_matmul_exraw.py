from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_matmul_exraw():

    def dm(d):
        result = {}
        for i, row in d.items():
            row = {j: val for j, val in row.items() if val}
            if row:
                result[i] = row
        return SDM(result, (2, 2), EXRAW)
    values = [S.NegativeInfinity, S.NegativeOne, S.Zero, S.One, S.Infinity]
    for a, b, c, d in product(*[values] * 4):
        Ad = dm({0: {0: a, 1: b}, 1: {0: c, 1: d}})
        Ad2 = dm({0: {0: a * a + b * c, 1: a * b + b * d}, 1: {0: c * a + d * c, 1: c * b + d * d}})
        assert Ad * Ad == Ad2