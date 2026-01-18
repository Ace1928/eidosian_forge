import numpy as np
def logdet_symm(m, check_symm=False):
    """
    Return log(det(m)) asserting positive definiteness of m.

    Parameters
    ----------
    m : array_like
        2d array that is positive-definite (and symmetric)

    Returns
    -------
    logdet : float
        The log-determinant of m.
    """
    from scipy import linalg
    if check_symm:
        if not np.all(m == m.T):
            raise ValueError('m is not symmetric.')
    c, _ = linalg.cho_factor(m, lower=True)
    return 2 * np.sum(np.log(c.diagonal()))