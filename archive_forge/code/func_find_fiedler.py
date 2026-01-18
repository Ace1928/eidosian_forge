from functools import partial
import networkx as nx
from networkx.utils import (
def find_fiedler(L, x, normalized, tol, seed):
    import scipy as sp
    L = sp.sparse.csc_array(L, dtype=float)
    n = L.shape[0]
    if normalized:
        D = sp.sparse.csc_array(sp.sparse.spdiags(1.0 / np.sqrt(L.diagonal()), [0], n, n, format='csc'))
        L = D @ L @ D
    if method == 'lanczos' or n < 10:
        sigma, X = sp.sparse.linalg.eigsh(L, 2, which='SM', tol=tol, return_eigenvectors=True)
        return (sigma[1], X[:, 1])
    else:
        X = np.asarray(np.atleast_2d(x).T)
        M = sp.sparse.csr_array(sp.sparse.spdiags(1.0 / L.diagonal(), 0, n, n))
        Y = np.ones(n)
        if normalized:
            Y /= D.diagonal()
        sigma, X = sp.sparse.linalg.lobpcg(L, X, M=M, Y=np.atleast_2d(Y).T, tol=tol, maxiter=n, largest=False)
        return (sigma[0], X[:, 0])