import numpy as np
from statsmodels.compat.pandas import Appender, Substitution
@Substitution(scale='4', extra_params='', extra_returns='', equation_number='9', equation='1/(4*d_j*d_k) * ((f(x + d[j]*e[j] + d[k]*e[k]) - f(x + d[j]*e[j]\n                                                     - d[k]*e[k])) -\n                 (f(x - d[j]*e[j] + d[k]*e[k]) - f(x - d[j]*e[j]\n                                                     - d[k]*e[k]))')
@Appender(_hessian_docs)
def approx_hess3(x, f, epsilon=None, args=(), kwargs={}):
    n = len(x)
    h = _get_epsilon(x, 4, epsilon, n)
    ee = np.diag(h)
    hess = np.outer(h, h)
    for i in range(n):
        for j in range(i, n):
            hess[i, j] = np.squeeze((f(*(x + ee[i, :] + ee[j, :],) + args, **kwargs) - f(*(x + ee[i, :] - ee[j, :],) + args, **kwargs) - (f(*(x - ee[i, :] + ee[j, :],) + args, **kwargs) - f(*(x - ee[i, :] - ee[j, :],) + args, **kwargs))) / (4.0 * hess[i, j]))
            hess[j, i] = hess[i, j]
    return hess