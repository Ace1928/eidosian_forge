from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyclasses import ANP
from sympy.polys.polyerrors import CoercionFailed, DomainError, NotAlgebraic, IsomorphismFailed
from sympy.utilities import public
def primes_above(self, p):
    """Compute the prime ideals lying above a given rational prime *p*."""
    from sympy.polys.numberfields.primes import prime_decomp
    ZK = self.maximal_order()
    dK = self.discriminant()
    rad = self._nilradicals_mod_p.get(p)
    return prime_decomp(p, ZK=ZK, dK=dK, radical=rad)