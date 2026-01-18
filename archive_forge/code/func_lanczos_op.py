import numpy as np
from scipy import sparse
from pygsp import utils
def lanczos_op(f, s, order=30):
    """
    Perform the lanczos approximation of the signal s.

    Parameters
    ----------
    f: Filter
    s : ndarray
        Signal to approximate.
    order : int
        Degree of the lanczos approximation. (default = 30)

    Returns
    -------
    L : ndarray
        lanczos approximation of s

    """
    G = f.G
    Nf = len(f.g)
    try:
        Nv = np.shape(s)[1]
        is2d = True
        c = np.zeros((G.N * Nf, Nv))
    except IndexError:
        Nv = 1
        is2d = False
        c = np.zeros(G.N * Nf)
    tmpN = np.arange(G.N, dtype=int)
    for j in range(Nv):
        if is2d:
            V, H, _ = lanczos(G.L.toarray(), order, s[:, j])
        else:
            V, H, _ = lanczos(G.L.toarray(), order, s)
        Eh, Uh = np.linalg.eig(H)
        Eh[Eh < 0] = 0
        fe = f.evaluate(Eh)
        V = np.dot(V, Uh)
        for i in range(Nf):
            if is2d:
                c[tmpN + i * G.N, j] = np.dot(V, fe[:][i] * np.dot(V.T, s[:, j]))
            else:
                c[tmpN + i * G.N] = np.dot(V, fe[:][i] * np.dot(V.T, s))
    return c