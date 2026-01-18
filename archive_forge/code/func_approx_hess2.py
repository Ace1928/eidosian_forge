import numpy as np
from statsmodels.compat.pandas import Appender, Substitution
@Substitution(scale='3', extra_params='return_grad : bool\n        Whether or not to also return the gradient\n', extra_returns='grad : ndarray\n        Gradient if return_grad == True\n', equation_number='8', equation='1/(2*d_j*d_k) * ((f(x + d[j]*e[j] + d[k]*e[k]) - f(x + d[j]*e[j])) -\n                 (f(x + d[k]*e[k]) - f(x)) +\n                 (f(x - d[j]*e[j] - d[k]*e[k]) - f(x + d[j]*e[j])) -\n                 (f(x - d[k]*e[k]) - f(x)))\n')
@Appender(_hessian_docs)
def approx_hess2(x, f, epsilon=None, args=(), kwargs={}, return_grad=False):
    n = len(x)
    h = _get_epsilon(x, 3, epsilon, n)
    ee = np.diag(h)
    f0 = f(*(x,) + args, **kwargs)
    g = np.zeros(n)
    gg = np.zeros(n)
    for i in range(n):
        g[i] = f(*(x + ee[i, :],) + args, **kwargs)
        gg[i] = f(*(x - ee[i, :],) + args, **kwargs)
    hess = np.outer(h, h)
    for i in range(n):
        for j in range(i, n):
            hess[i, j] = (f(*(x + ee[i, :] + ee[j, :],) + args, **kwargs) - g[i] - g[j] + f0 + f(*(x - ee[i, :] - ee[j, :],) + args, **kwargs) - gg[i] - gg[j] + f0) / (2 * hess[i, j])
            hess[j, i] = hess[i, j]
    if return_grad:
        grad = (g - f0) / h
        return (hess, grad)
    else:
        return hess