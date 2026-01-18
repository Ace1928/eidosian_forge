from __future__ import division
def bateman_parent(lmbd, t, one=1, zero=0, exp=None):
    n = len(lmbd)
    N = [None] * n
    lmbd_prod = one
    if exp is None:
        import math
        exp = math.exp
    for i in range(n):
        if i > 0:
            lmbd_prod *= lmbd[i - 1]
        sum_k = zero
        for k in range(i + 1):
            prod_l = one
            for l in range(i + 1):
                if l == k:
                    continue
                prod_l *= lmbd[l] - lmbd[k]
            sum_k += exp(-lmbd[k] * t) / prod_l
        N[i] = lmbd_prod * sum_k
    return N