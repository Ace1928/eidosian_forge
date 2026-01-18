import itertools
import math
import numpy as np
def moments_raw_to_central(moments_raw):
    ndim = moments_raw.ndim
    order = moments_raw.shape[0] - 1
    if ndim in [2, 3] and order < 4:
        return _moments_raw_to_central_fast(moments_raw)
    moments_central = np.zeros_like(moments_raw)
    m = moments_raw
    centers = tuple(m[tuple(np.eye(ndim, dtype=int))] / m[(0,) * ndim])
    if ndim == 2:
        for p in range(order + 1):
            for q in range(order + 1):
                if p + q > order:
                    continue
                for i in range(p + 1):
                    term1 = math.comb(p, i)
                    term1 *= (-centers[0]) ** (p - i)
                    for j in range(q + 1):
                        term2 = math.comb(q, j)
                        term2 *= (-centers[1]) ** (q - j)
                        moments_central[p, q] += term1 * term2 * m[i, j]
        return moments_central
    for orders in itertools.product(*(range(order + 1),) * ndim):
        if sum(orders) > order:
            continue
        for idxs in itertools.product(*[range(o + 1) for o in orders]):
            val = m[idxs]
            for i_order, c, idx in zip(orders, centers, idxs):
                val *= math.comb(i_order, idx)
                val *= (-c) ** (i_order - idx)
            moments_central[orders] += val
    return moments_central