from __future__ import absolute_import
from functools import partial
import numpy.linalg as npla
from .numpy_wrapper import wrap_namespace
from . import numpy_wrapper as anp
from autograd.extend import defvjp, defjvp
def grad_eigh(ans, x, UPLO='L'):
    """Gradient for eigenvalues and vectors of a symmetric matrix."""
    N = x.shape[-1]
    w, v = ans
    vc = anp.conj(v)

    def vjp(g):
        wg, vg = g
        w_repeated = anp.repeat(w[..., anp.newaxis], N, axis=-1)
        vjp_temp = _dot(vc * wg[..., anp.newaxis, :], T(v))
        if anp.any(vg):
            off_diag = anp.ones((N, N)) - anp.eye(N)
            F = off_diag / (T(w_repeated) - w_repeated + anp.eye(N))
            vjp_temp += _dot(_dot(vc, F * _dot(T(v), vg)), T(v))
        reps = anp.array(x.shape)
        reps[-2:] = 1
        if UPLO == 'L':
            tri = anp.tile(anp.tril(anp.ones(N), -1), reps)
        elif UPLO == 'U':
            tri = anp.tile(anp.triu(anp.ones(N), 1), reps)
        return anp.real(vjp_temp) * anp.eye(vjp_temp.shape[-1]) + (vjp_temp + anp.conj(T(vjp_temp))) * tri
    return vjp