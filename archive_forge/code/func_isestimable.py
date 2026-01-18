import numpy as np
import pandas as pd
import scipy.linalg
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import array_like
def isestimable(c, d):
    """
    True if (Q, P) contrast `c` is estimable for (N, P) design `d`.

    From an Q x P contrast matrix `C` and an N x P design matrix `D`, checks if
    the contrast `C` is estimable by looking at the rank of ``vstack([C,D])``
    and verifying it is the same as the rank of `D`.

    Parameters
    ----------
    c : array_like
        A contrast matrix with shape (Q, P). If 1 dimensional assume shape is
        (1, P).
    d : array_like
        The design matrix, (N, P).

    Returns
    -------
    bool
        True if the contrast `c` is estimable on design `d`.

    Examples
    --------
    >>> d = np.array([[1, 1, 1, 0, 0, 0],
    ...               [0, 0, 0, 1, 1, 1],
    ...               [1, 1, 1, 1, 1, 1]]).T
    >>> isestimable([1, 0, 0], d)
    False
    >>> isestimable([1, -1, 0], d)
    True
    """
    c = array_like(c, 'c', maxdim=2)
    d = array_like(d, 'd', ndim=2)
    c = c[None, :] if c.ndim == 1 else c
    if c.shape[1] != d.shape[1]:
        raise ValueError('Contrast should have %d columns' % d.shape[1])
    new = np.vstack([c, d])
    if np.linalg.matrix_rank(new) != np.linalg.matrix_rank(d):
        return False
    return True