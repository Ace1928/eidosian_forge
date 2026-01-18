from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
def dmp_norm(f, u, K):
    """
    Norm of ``f`` in ``K[X1, ..., Xn]``, often not square-free.
    """
    if not K.is_Algebraic:
        raise DomainError('ground domain must be algebraic')
    g = dmp_raise(K.mod.rep, u + 1, 0, K.dom)
    h, _ = dmp_inject(f, u, K, front=True)
    return dmp_resultant(g, h, u + 1, K.dom)