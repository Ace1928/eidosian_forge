from sympy.core.random import _randint
from sympy.polys.galoistools import (
from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.sqfreetools import (
from sympy.polys.polyutils import _sort_factors
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import (
from sympy.utilities import subsets
from math import ceil as _ceil, log as _log
def dmp_zz_wang_test_points(f, T, ct, A, u, K):
    """Wang/EEZ: Test evaluation points for suitability. """
    if not dmp_eval_tail(dmp_LC(f, K), A, u - 1, K):
        raise EvaluationFailed('no luck')
    g = dmp_eval_tail(f, A, u, K)
    if not dup_sqf_p(g, K):
        raise EvaluationFailed('no luck')
    c, h = dup_primitive(g, K)
    if K.is_negative(dup_LC(h, K)):
        c, h = (-c, dup_neg(h, K))
    v = u - 1
    E = [dmp_eval_tail(t, A, v, K) for t, _ in T]
    D = dmp_zz_wang_non_divisors(E, c, ct, K)
    if D is not None:
        return (c, h, E)
    else:
        raise EvaluationFailed('no luck')