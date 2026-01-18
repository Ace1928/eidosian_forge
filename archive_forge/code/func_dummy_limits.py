from numpy.testing import assert_equal
import numpy as np
def dummy_limits(d):
    """start and endpoints of groups in a sorted dummy variable array

    helper function for nested categories

    Examples
    --------
    >>> d1 = np.array([[1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 1, 0],
                       [0, 1, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 1],
                       [0, 0, 1],
                       [0, 0, 1]])
    >>> dummy_limits(d1)
    (array([0, 4, 8]), array([ 4,  8, 12]))

    get group slices from an array

    >>> [np.arange(d1.shape[0])[b:e] for b,e in zip(*dummy_limits(d1))]
    [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([ 8,  9, 10, 11])]
    >>> [np.arange(d1.shape[0])[b:e] for b,e in zip(*dummy_limits(d1))]
    [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([ 8,  9, 10, 11])]
    """
    nobs, nvars = d.shape
    start1, col1 = np.nonzero(np.diff(d, axis=0) == 1)
    end1, col1_ = np.nonzero(np.diff(d, axis=0) == -1)
    cc = np.arange(nvars)
    if not (np.r_[[0], col1] == cc).all() or not (np.r_[col1_, [nvars - 1]] == cc).all():
        raise ValueError('dummy variable is not sorted')
    start = np.r_[[0], start1 + 1]
    end = np.r_[end1 + 1, [nobs]]
    return (start, end)