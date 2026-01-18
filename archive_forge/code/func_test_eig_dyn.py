from mpmath import mp
from mpmath import libmp
def test_eig_dyn():
    v = 0
    for i in xrange(5):
        n = 1 + int(mp.rand() * 5)
        if mp.rand() > 0.5:
            A = 2 * mp.randmatrix(n, n) - 1
            if mp.rand() > 0.5:
                A *= 10
                for x in xrange(n):
                    for y in xrange(n):
                        A[x, y] = int(A[x, y])
        else:
            A = 2 * mp.randmatrix(n, n) - 1 + 1j * (2 * mp.randmatrix(n, n) - 1)
            if mp.rand() > 0.5:
                A *= 10
                for x in xrange(n):
                    for y in xrange(n):
                        A[x, y] = int(mp.re(A[x, y])) + 1j * int(mp.im(A[x, y]))
        run_hessenberg(A, verbose=v)
        run_schur(A, verbose=v)
        run_eig(A, verbose=v)