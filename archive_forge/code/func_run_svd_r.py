from mpmath import mp
from mpmath import libmp
def run_svd_r(A, full_matrices=False, verbose=True):
    m, n = (A.rows, A.cols)
    eps = mp.exp(0.8 * mp.log(mp.eps))
    if verbose:
        print('original matrix:\n', str(A))
        print('full', full_matrices)
    U, S0, V = mp.svd_r(A, full_matrices=full_matrices)
    S = mp.zeros(U.cols, V.rows)
    for j in xrange(min(m, n)):
        S[j, j] = S0[j]
    if verbose:
        print('U:\n', str(U))
        print('S:\n', str(S0))
        print('V:\n', str(V))
    C = U * S * V - A
    err = mp.mnorm(C)
    if verbose:
        print('C\n', str(C), '\n', err)
    assert err < eps
    D = V * V.transpose() - mp.eye(V.rows)
    err = mp.mnorm(D)
    if verbose:
        print('D:\n', str(D), '\n', err)
    assert err < eps
    E = U.transpose() * U - mp.eye(U.cols)
    err = mp.mnorm(E)
    if verbose:
        print('E:\n', str(E), '\n', err)
    assert err < eps