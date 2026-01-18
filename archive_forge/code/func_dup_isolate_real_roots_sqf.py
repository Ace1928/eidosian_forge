from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def dup_isolate_real_roots_sqf(f, K, eps=None, inf=None, sup=None, fast=False, blackbox=False):
    """Isolate real roots of a square-free polynomial using the Vincent-Akritas-Strzebonski (VAS) CF approach.

       References
       ==========
       .. [1] Alkiviadis G. Akritas and Adam W. Strzebonski: A Comparative
              Study of Two Real Root Isolation Methods. Nonlinear Analysis:
              Modelling and Control, Vol. 10, No. 4, 297-304, 2005.
       .. [2] Alkiviadis G. Akritas, Adam W. Strzebonski and Panagiotis S.
              Vigklas: Improving the Performance of the Continued Fractions
              Method Using New Bounds of Positive Roots. Nonlinear Analysis:
              Modelling and Control, Vol. 13, No. 3, 265-279, 2008.

    """
    if K.is_QQ:
        (_, f), K = (dup_clear_denoms(f, K, convert=True), K.get_ring())
    elif not K.is_ZZ:
        raise DomainError('isolation of real roots not supported over %s' % K)
    if dup_degree(f) <= 0:
        return []
    I_zero, f = _isolate_zero(f, K, inf, sup, basis=False, sqf=True)
    I_neg = dup_inner_isolate_negative_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast)
    I_pos = dup_inner_isolate_positive_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast)
    roots = sorted(I_neg + I_zero + I_pos)
    if not blackbox:
        return roots
    else:
        return [RealInterval((a, b), f, K) for a, b in roots]