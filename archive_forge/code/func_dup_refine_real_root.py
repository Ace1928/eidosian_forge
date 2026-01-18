from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def dup_refine_real_root(f, s, t, K, eps=None, steps=None, disjoint=None, fast=False):
    """Refine real root's approximating interval to the given precision. """
    if K.is_QQ:
        (_, f), K = (dup_clear_denoms(f, K, convert=True), K.get_ring())
    elif not K.is_ZZ:
        raise DomainError('real root refinement not supported over %s' % K)
    if s == t:
        return (s, t)
    if s > t:
        s, t = (t, s)
    negative = False
    if s < 0:
        if t <= 0:
            f, s, t, negative = (dup_mirror(f, K), -t, -s, True)
        else:
            raise ValueError('Cannot refine a real root in (%s, %s)' % (s, t))
    if negative and disjoint is not None:
        if disjoint < 0:
            disjoint = -disjoint
        else:
            disjoint = None
    s, t = dup_outer_refine_real_root(f, s, t, K, eps=eps, steps=steps, disjoint=disjoint, fast=fast)
    if negative:
        return (-t, -s)
    else:
        return (s, t)