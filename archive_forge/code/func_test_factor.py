from sympy.polys.polytools import Poly
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
from sympy.utilities.decorator import public
from .basis import round_two, nilradical_mod_p
from .exceptions import StructureError
from .modules import ModuleEndomorphism, find_min_poly
from .utilities import coeff_search, supplement_a_subspace
def test_factor(self):
    """
        Compute a test factor for this prime ideal.

        Explanation
        ===========

        Write $\\mathfrak{p}$ for this prime ideal, $p$ for the rational prime
        it divides. Then, for computing $\\mathfrak{p}$-adic valuations it is
        useful to have a number $\\beta \\in \\mathbb{Z}_K$ such that
        $p/\\mathfrak{p} = p \\mathbb{Z}_K + \\beta \\mathbb{Z}_K$.

        Essentially, this is the same as the number $\\Psi$ (or the "reagent")
        from Kummer's 1847 paper (*Ueber die Zerlegung...*, Crelle vol. 35) in
        which ideal divisors were invented.
        """
    if self._test_factor is None:
        self._test_factor = _compute_test_factor(self.p, [self.alpha], self.ZK)
    return self._test_factor