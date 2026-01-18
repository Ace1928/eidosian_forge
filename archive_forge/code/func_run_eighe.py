from mpmath import mp
from mpmath import libmp
def run_eighe(A, verbose=False):
    if verbose:
        print('original matrix:\n', str(A))
    D, Q = mp.eighe(A)
    B = Q * mp.diag(D) * Q.transpose_conj()
    C = A - B
    E = Q * Q.transpose_conj() - mp.eye(A.rows)
    if verbose:
        print('eigenvalues:\n', D)
        print('eigenvectors:\n', Q)
    NC = mp.mnorm(C)
    NE = mp.mnorm(E)
    if verbose:
        print('difference:', NC, '\n', C, '\n')
        print('difference:', NE, '\n', E, '\n')
    eps = mp.exp(0.8 * mp.log(mp.eps))
    assert NC < eps
    assert NE < eps
    return NC