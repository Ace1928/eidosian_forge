import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as student_t
from scipy import stats
from statsmodels.tools.tools import clean0, fullrank
from statsmodels.stats.multitest import multipletests
def contrastfromcols(L, D, pseudo=None):
    """
    From an n x p design matrix D and a matrix L, tries
    to determine a p x q contrast matrix C which
    determines a contrast of full rank, i.e. the
    n x q matrix

    dot(transpose(C), pinv(D))

    is full rank.

    L must satisfy either L.shape[0] == n or L.shape[1] == p.

    If L.shape[0] == n, then L is thought of as representing
    columns in the column space of D.

    If L.shape[1] == p, then L is thought of as what is known
    as a contrast matrix. In this case, this function returns an estimable
    contrast corresponding to the dot(D, L.T)

    Note that this always produces a meaningful contrast, not always
    with the intended properties because q is always non-zero unless
    L is identically 0. That is, it produces a contrast that spans
    the column space of L (after projection onto the column space of D).

    Parameters
    ----------
    L : array_like
    D : array_like
    """
    L = np.asarray(L)
    D = np.asarray(D)
    n, p = D.shape
    if L.shape[0] != n and L.shape[1] != p:
        raise ValueError('shape of L and D mismatched')
    if pseudo is None:
        pseudo = np.linalg.pinv(D)
    if L.shape[0] == n:
        C = np.dot(pseudo, L).T
    else:
        C = L
        C = np.dot(pseudo, np.dot(D, C.T)).T
    Lp = np.dot(D, C.T)
    if len(Lp.shape) == 1:
        Lp.shape = (n, 1)
    if np.linalg.matrix_rank(Lp) != Lp.shape[1]:
        Lp = fullrank(Lp)
        C = np.dot(pseudo, Lp).T
    return np.squeeze(C)