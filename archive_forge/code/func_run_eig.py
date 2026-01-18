from mpmath import mp
from mpmath import libmp
def run_eig(A, verbose=0):
    if verbose > 1:
        print('original matrix (eig):\n', A)
    n = A.rows
    E, EL, ER = mp.eig(A, left=True, right=True)
    if verbose > 1:
        print('E:\n', E)
        print('EL:\n', EL)
        print('ER:\n', ER)
    eps = mp.exp(0.8 * mp.log(mp.eps))
    err0 = 0
    for i in xrange(n):
        B = A * ER[:, i] - E[i] * ER[:, i]
        err0 = max(err0, mp.mnorm(B))
        B = EL[i, :] * A - EL[i, :] * E[i]
        err0 = max(err0, mp.mnorm(B))
    err0 /= n * n
    if verbose > 0:
        print('difference (E):', err0)
    assert err0 < eps