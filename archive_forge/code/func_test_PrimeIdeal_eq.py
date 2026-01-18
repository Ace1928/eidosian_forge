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
def test_PrimeIdeal_eq():
    T = Poly(cyclotomic_poly(7))
    P0 = prime_decomp(5, T)[0]
    assert P0.f == 6
    assert P0.as_submodule() == 5 * P0.ZK
    assert P0 != 5