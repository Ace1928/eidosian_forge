from __future__ import division
def bateman_full(y0s, lmbd, t, one=1, zero=0, exp=None):
    n = len(lmbd)
    if len(y0s) != n:
        raise ValueError('Please pass equal number of decay constants as initial concentrations (you may want to pad lmbd with zeroes)')
    N = [zero] * n
    for i, y0 in enumerate(y0s):
        if y0 == zero:
            continue
        Ni = bateman_parent(lmbd[i:], t, one, zero, exp)
        for j, yj in enumerate(Ni, i):
            N[j] += y0 * yj
    return N