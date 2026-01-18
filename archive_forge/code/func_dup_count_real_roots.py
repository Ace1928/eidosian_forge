from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def dup_count_real_roots(f, K, inf=None, sup=None):
    """Returns the number of distinct real roots of ``f`` in ``[inf, sup]``. """
    if dup_degree(f) <= 0:
        return 0
    if not K.is_Field:
        R, K = (K, K.get_field())
        f = dup_convert(f, R, K)
    sturm = dup_sturm(f, K)
    if inf is None:
        signs_inf = dup_sign_variations([dup_LC(s, K) * (-1) ** dup_degree(s) for s in sturm], K)
    else:
        signs_inf = dup_sign_variations([dup_eval(s, inf, K) for s in sturm], K)
    if sup is None:
        signs_sup = dup_sign_variations([dup_LC(s, K) for s in sturm], K)
    else:
        signs_sup = dup_sign_variations([dup_eval(s, sup, K) for s in sturm], K)
    count = abs(signs_inf - signs_sup)
    if inf is not None and (not dup_eval(f, inf, K)):
        count += 1
    return count