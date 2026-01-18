import osqp
import numpy as np
import numpy.random as npr
from scipy import sparse
from scipy.optimize import approx_fprime
import numpy.testing as npt
import unittest
def test_dl_dl(self, verbose=False):
    n, m = (10, 10)
    prob = self.get_prob(n=n, m=m, P_scale=100.0, A_scale=100.0)
    P, q, A, l, u, true_x = prob

    def grad(l):
        [dP, dq, dA, dl, du] = self.get_grads(P, q, A, l, u, true_x)
        return dl

    def f(l):
        m = osqp.OSQP()
        m.setup(P, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel, max_iter=max_iter, verbose=False)
        res = m.solve()
        if res.info.status != 'solved':
            raise ValueError('Problem not solved!')
        x_hat = res.x
        return 0.5 * np.sum(np.square(x_hat - true_x))
    dl = grad(l)
    dl_fd = approx_fprime(l, f, grad_precision)
    if verbose:
        print('dl_fd: ', np.round(dl_fd, decimals=4))
        print('dl: ', np.round(dl, decimals=4))
    npt.assert_allclose(dl_fd, dl, rtol=rel_tol, atol=abs_tol)