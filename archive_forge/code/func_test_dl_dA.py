import osqp
import numpy as np
import numpy.random as npr
from scipy import sparse
from scipy.optimize import approx_fprime
import numpy.testing as npt
import unittest
def test_dl_dA(self, verbose=False):
    n, m = (3, 3)
    prob = self.get_prob(n=n, m=m, P_scale=100.0, A_scale=100.0)
    P, q, A, l, u, true_x = prob
    A_idx = A.nonzero()

    def grad(A_val):
        A_qp = sparse.csc_matrix((A_val, A_idx), shape=A.shape)
        [dP, dq, dA, dl, du] = self.get_grads(P, q, A_qp, l, u, true_x)
        return dA

    def f(A_val):
        A_qp = sparse.csc_matrix((A_val, A_idx), shape=A.shape)
        m = osqp.OSQP()
        m.setup(P, q, A_qp, l, u, eps_abs=eps_abs, eps_rel=eps_rel, max_iter=max_iter, verbose=False)
        res = m.solve()
        if res.info.status != 'solved':
            raise ValueError('Problem not solved!')
        x_hat = res.x
        return 0.5 * np.sum(np.square(x_hat - true_x))
    dA = grad(A.data)
    dA_fd_val = approx_fprime(A.data, f, grad_precision)
    dA_fd = sparse.csc_matrix((dA_fd_val, A_idx), shape=A.shape)
    if verbose:
        print('dA_fd: ', np.round(dA_fd.data, decimals=4))
        print('dA: ', np.round(dA.data, decimals=4))
    npt.assert_allclose(dA.todense(), dA_fd.todense(), rtol=rel_tol, atol=abs_tol)