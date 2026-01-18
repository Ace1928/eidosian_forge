from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dmp_clear_denoms(f, u, K0, K1=None, convert=False):
    """
    Clear denominators, i.e. transform ``K_0`` to ``K_1``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    >>> f = QQ(1,2)*x + QQ(1,3)*y + 1

    >>> R.dmp_clear_denoms(f, convert=False)
    (6, 3*x + 2*y + 6)
    >>> R.dmp_clear_denoms(f, convert=True)
    (6, 3*x + 2*y + 6)

    """
    if not u:
        return dup_clear_denoms(f, K0, K1, convert=convert)
    if K1 is None:
        if K0.has_assoc_Ring:
            K1 = K0.get_ring()
        else:
            K1 = K0
    common = _rec_clear_denoms(f, u, K0, K1)
    if not K1.is_one(common):
        f = dmp_mul_ground(f, common, u, K0)
    if not convert:
        return (common, f)
    else:
        return (common, dmp_convert(f, u, K0, K1))