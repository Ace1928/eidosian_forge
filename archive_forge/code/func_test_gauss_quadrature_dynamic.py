from mpmath import mp
from mpmath import libmp
def test_gauss_quadrature_dynamic(verbose=False):
    n = 5
    A = mp.randmatrix(2 * n, 1)

    def F(x):
        r = 0
        for i in xrange(len(A) - 1, -1, -1):
            r = r * x + A[i]
        return r

    def run(qtype, FW, R, alpha=0, beta=0):
        X, W = mp.gauss_quadrature(n, qtype, alpha=alpha, beta=beta)
        a = 0
        for i in xrange(len(X)):
            a += W[i] * F(X[i])
        b = mp.quad(lambda x: FW(x) * F(x), R)
        c = mp.fabs(a - b)
        if verbose:
            print(qtype, c, a, b)
        assert c < 1e-05
    run('legendre', lambda x: 1, [-1, 1])
    run('legendre01', lambda x: 1, [0, 1])
    run('hermite', lambda x: mp.exp(-x * x), [-mp.inf, mp.inf])
    run('laguerre', lambda x: mp.exp(-x), [0, mp.inf])
    run('glaguerre', lambda x: mp.sqrt(x) * mp.exp(-x), [0, mp.inf], alpha=1 / mp.mpf(2))
    run('chebyshev1', lambda x: 1 / mp.sqrt(1 - x * x), [-1, 1])
    run('chebyshev2', lambda x: mp.sqrt(1 - x * x), [-1, 1])
    run('jacobi', lambda x: (1 - x) ** (1 / mp.mpf(3)) * (1 + x) ** (1 / mp.mpf(5)), [-1, 1], alpha=1 / mp.mpf(3), beta=1 / mp.mpf(5))