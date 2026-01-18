from scipy import stats, integrate, special
import numpy as np
def density_orthopoly(x, polybase, order=5, xeval=None):
    if xeval is None:
        xeval = np.linspace(x.min(), x.max(), 50)
    polys = [polybase(i) for i in range(order)]
    coeffs = [p(x).mean() for p in polys]
    res = sum((c * p(xeval) for c, p in zip(coeffs, polys)))
    return (res, xeval, coeffs, polys)