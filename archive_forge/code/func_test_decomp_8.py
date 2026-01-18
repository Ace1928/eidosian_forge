from math import prod
from sympy import QQ, ZZ
from sympy.abc import x, theta
from sympy.ntheory import factorint
from sympy.ntheory.residue_ntheory import n_order
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.matrices import DomainMatrix
from sympy.polys.numberfields.basis import round_two
from sympy.polys.numberfields.exceptions import StructureError
from sympy.polys.numberfields.modules import PowerBasis, to_col
from sympy.polys.numberfields.primes import (
from sympy.testing.pytest import raises
def test_decomp_8():
    cases = (x ** 3 + 3 * x ** 2 - 4 * x + 4, x ** 3 + 3 * x ** 2 + 3 * x - 3, x ** 3 + 5 * x ** 2 - x + 3, x ** 3 + 5 * x ** 2 - 5 * x - 5, x ** 3 + 3 * x ** 2 + 5, x ** 3 + 6 * x ** 2 + 3 * x - 1, x ** 3 + 6 * x ** 2 + 4, x ** 3 + 7 * x ** 2 + 7 * x - 7, x ** 3 + 7 * x ** 2 - x + 5, x ** 3 + 7 * x ** 2 - 5 * x + 5, x ** 3 + 4 * x ** 2 - 3 * x + 7, x ** 3 + 8 * x ** 2 + 5 * x - 1, x ** 3 + 8 * x ** 2 - 2 * x + 6, x ** 3 + 6 * x ** 2 - 3 * x + 8, x ** 3 + 9 * x ** 2 + 6 * x - 8, x ** 3 + 15 * x ** 2 - 9 * x + 13)

    def display(T, p, radical, P, I, J):
        """Useful for inspection, when running test manually."""
        print('=' * 20)
        print(T, p, radical)
        for Pi in P:
            print(f'  ({Pi!r})')
        print('I: ', I)
        print('J: ', J)
        print(f'Equal: {I == J}')
    inspect = False
    for g in cases:
        T = Poly(g)
        rad = {}
        ZK, dK = round_two(T, radicals=rad)
        dT = T.discriminant()
        f_squared = dT // dK
        F = factorint(f_squared)
        for p in F:
            radical = rad.get(p)
            P = prime_decomp(p, T, dK=dK, ZK=ZK, radical=radical)
            I = prod((Pi ** Pi.e for Pi in P))
            J = p * ZK
            if inspect:
                display(T, p, radical, P, I, J)
            assert I == J