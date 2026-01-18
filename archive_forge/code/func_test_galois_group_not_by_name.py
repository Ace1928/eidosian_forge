from sympy.abc import x
from sympy.combinatorics.galois import (
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.numberfields.galoisgroups import (
from sympy.polys.numberfields.subfield import field_isomorphism
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
def test_galois_group_not_by_name():
    """
    Check at least one polynomial of each supported degree, to see that
    conversion from name to group works.
    """
    for deg in range(1, 7):
        T, G_name, _ = test_polys_by_deg[deg][0]
        G, _ = galois_group(T)
        assert G == G_name.get_perm_group()