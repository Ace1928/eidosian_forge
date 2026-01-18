from numpy.testing import assert_equal
import numpy as np
def dummy_nested(d1, d2, method='full'):
    """unfinished and incomplete mainly copy past dummy_product
    dummy variable from product of two dummy variables

    Parameters
    ----------
    d1, d2 : ndarray
        two dummy variables, d2 is assumed to be nested in d1
        Assumes full set for methods 'drop-last' and 'drop-first'.
    method : {'full', 'drop-last', 'drop-first'}
        'full' returns the full product, which in this case is d2.
        The drop methods provide an effects encoding:
        (constant, main effects, subgroup effects). The first or last columns
        of the dummy variable (i.e. levels) are dropped to get full rank
        encoding.

    Returns
    -------
    dummy : ndarray
        dummy variable for product, see method

    """
    if method == 'full':
        return d2
    start1, end1 = dummy_limits(d1)
    start2, end2 = dummy_limits(d2)
    first = np.in1d(start2, start1)
    last = np.in1d(end2, end1)
    equal = first == last
    col_dropf = ~first * ~equal
    col_dropl = ~last * ~equal
    if method == 'drop-last':
        d12rl = dummy_product(d1[:, :-1], d2[:, :-1])
        dd = np.column_stack((np.ones(d1.shape[0], int), d1[:, :-1], d2[:, col_dropl]))
    elif method == 'drop-first':
        d12r = dummy_product(d1[:, 1:], d2[:, 1:])
        dd = np.column_stack((np.ones(d1.shape[0], int), d1[:, 1:], d2[:, col_dropf]))
    else:
        raise ValueError('method not recognized')
    return (dd, col_dropf, col_dropl)