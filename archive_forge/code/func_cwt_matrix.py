import numpy as np
from scipy._lib._util import check_random_state, rng_integers
from scipy.sparse import csc_matrix
def cwt_matrix(n_rows, n_columns, seed=None):
    """
    Generate a matrix S which represents a Clarkson-Woodruff transform.

    Given the desired size of matrix, the method returns a matrix S of size
    (n_rows, n_columns) where each column has all the entries set to 0
    except for one position which has been randomly set to +1 or -1 with
    equal probability.

    Parameters
    ----------
    n_rows : int
        Number of rows of S
    n_columns : int
        Number of columns of S
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    S : (n_rows, n_columns) csc_matrix
        The returned matrix has ``n_columns`` nonzero entries.

    Notes
    -----
    Given a matrix A, with probability at least 9/10,
    .. math:: \\|SA\\| = (1 \\pm \\epsilon)\\|A\\|
    Where the error epsilon is related to the size of S.
    """
    rng = check_random_state(seed)
    rows = rng_integers(rng, 0, n_rows, n_columns)
    cols = np.arange(n_columns + 1)
    signs = rng.choice([1, -1], n_columns)
    S = csc_matrix((signs, rows, cols), shape=(n_rows, n_columns))
    return S