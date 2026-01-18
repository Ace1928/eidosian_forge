from sympy.abc import x
from sympy.combinatorics.galois import (
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.numberfields.galoisgroups import (
from sympy.polys.numberfields.subfield import field_isomorphism
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
def test__galois_group_degree_4_root_approx():
    for T, G, alt in test_polys_by_deg[4]:
        assert _galois_group_degree_4_root_approx(Poly(T)) == (G, alt)