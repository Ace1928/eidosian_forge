from mpmath import mp
from mpmath import libmp
def run_hessenberg(A, verbose=0):
    if verbose > 1:
        print('original matrix (hessenberg):\n', A)
    n = A.rows
    Q, H = mp.hessenberg(A)
    if verbose > 1:
        print('Q:\n', Q)
        print('H:\n', H)
    B = Q * H * Q.transpose_conj()
    eps = mp.exp(0.8 * mp.log(mp.eps))
    err0 = 0
    for x in xrange(n):
        for y in xrange(n):
            err0 += abs(A[y, x] - B[y, x])
    err0 /= n * n
    err1 = 0
    for x in xrange(n):
        for y in xrange(x + 2, n):
            err1 += abs(H[y, x])
    if verbose > 0:
        print('difference (H):', err0, err1)
    if verbose > 1:
        print('B:\n', B)
    assert err0 < eps
    assert err1 == 0