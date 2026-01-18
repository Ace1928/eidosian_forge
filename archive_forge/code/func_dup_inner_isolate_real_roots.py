from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def dup_inner_isolate_real_roots(f, K, eps=None, fast=False):
    """Internal function for isolation positive roots up to given precision.

       References
       ==========
           1. Alkiviadis G. Akritas and Adam W. Strzebonski: A Comparative Study of Two Real Root
           Isolation Methods . Nonlinear Analysis: Modelling and Control, Vol. 10, No. 4, 297-304, 2005.
           2. Alkiviadis G. Akritas, Adam W. Strzebonski and Panagiotis S. Vigklas: Improving the
           Performance of the Continued Fractions Method Using new Bounds of Positive Roots. Nonlinear
           Analysis: Modelling and Control, Vol. 13, No. 3, 265-279, 2008.
    """
    a, b, c, d = (K.one, K.zero, K.zero, K.one)
    k = dup_sign_variations(f, K)
    if k == 0:
        return []
    if k == 1:
        roots = [dup_inner_refine_real_root(f, (a, b, c, d), K, eps=eps, fast=fast, mobius=True)]
    else:
        roots, stack = ([], [(a, b, c, d, f, k)])
        while stack:
            a, b, c, d, f, k = stack.pop()
            A = dup_root_lower_bound(f, K)
            if A is not None:
                A = K(int(A))
            else:
                A = K.zero
            if fast and A > 16:
                f = dup_scale(f, A, K)
                a, c, A = (A * a, A * c, K.one)
            if A >= K.one:
                f = dup_shift(f, A, K)
                b, d = (A * a + b, A * c + d)
                if not dup_TC(f, K):
                    roots.append((f, (b, b, d, d)))
                    f = dup_rshift(f, 1, K)
                k = dup_sign_variations(f, K)
                if k == 0:
                    continue
                if k == 1:
                    roots.append(dup_inner_refine_real_root(f, (a, b, c, d), K, eps=eps, fast=fast, mobius=True))
                    continue
            f1 = dup_shift(f, K.one, K)
            a1, b1, c1, d1, r = (a, a + b, c, c + d, 0)
            if not dup_TC(f1, K):
                roots.append((f1, (b1, b1, d1, d1)))
                f1, r = (dup_rshift(f1, 1, K), 1)
            k1 = dup_sign_variations(f1, K)
            k2 = k - k1 - r
            a2, b2, c2, d2 = (b, a + b, d, c + d)
            if k2 > 1:
                f2 = dup_shift(dup_reverse(f), K.one, K)
                if not dup_TC(f2, K):
                    f2 = dup_rshift(f2, 1, K)
                k2 = dup_sign_variations(f2, K)
            else:
                f2 = None
            if k1 < k2:
                a1, a2, b1, b2 = (a2, a1, b2, b1)
                c1, c2, d1, d2 = (c2, c1, d2, d1)
                f1, f2, k1, k2 = (f2, f1, k2, k1)
            if not k1:
                continue
            if f1 is None:
                f1 = dup_shift(dup_reverse(f), K.one, K)
                if not dup_TC(f1, K):
                    f1 = dup_rshift(f1, 1, K)
            if k1 == 1:
                roots.append(dup_inner_refine_real_root(f1, (a1, b1, c1, d1), K, eps=eps, fast=fast, mobius=True))
            else:
                stack.append((a1, b1, c1, d1, f1, k1))
            if not k2:
                continue
            if f2 is None:
                f2 = dup_shift(dup_reverse(f), K.one, K)
                if not dup_TC(f2, K):
                    f2 = dup_rshift(f2, 1, K)
            if k2 == 1:
                roots.append(dup_inner_refine_real_root(f2, (a2, b2, c2, d2), K, eps=eps, fast=fast, mobius=True))
            else:
                stack.append((a2, b2, c2, d2, f2, k2))
    return roots