from __future__ import division
import scipy.linalg
import autograd.numpy as anp
from autograd.numpy.numpy_wrapper import wrap_namespace
from autograd.extend import defvjp, defvjp_argnums, defjvp, defjvp_argnums
def grad_solve_triangular(ans, a, b, trans=0, lower=False, **kwargs):
    tri = anp.tril if lower ^ (_flip(a, trans) == 'N') else anp.triu
    transpose = lambda x: x if _flip(a, trans) != 'N' else x.T
    al2d = lambda x: x if x.ndim > 1 else x[..., None]

    def vjp(g):
        v = al2d(solve_triangular(a, g, trans=_flip(a, trans), lower=lower))
        return -transpose(tri(anp.dot(v, al2d(ans).T)))
    return vjp