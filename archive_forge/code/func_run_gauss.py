from mpmath import mp
from mpmath import libmp
def run_gauss(qtype, a, b):
    eps = 1e-05
    d, e = mp.gauss_quadrature(len(a), qtype)
    d -= mp.matrix(a)
    e -= mp.matrix(b)
    assert mp.mnorm(d) < eps
    assert mp.mnorm(e) < eps