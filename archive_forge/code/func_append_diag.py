from functools import reduce
import numpy as np
def append_diag(aff, steps, starts=()):
    """Add diagonal elements `steps` and translations `starts` to affine

    Typical use is in expanding 4x4 affines to larger dimensions.  Nipy is the
    main consumer because it uses NxM affines, whereas we generally only use
    4x4 affines; the routine is here for convenience.

    Parameters
    ----------
    aff : 2D array
        N by M affine matrix
    steps : scalar or sequence
        diagonal elements to append.
    starts : scalar or sequence
        elements to append to last column of `aff`, representing translations
        corresponding to the `steps`. If empty, expands to a vector of zeros
        of the same length as `steps`

    Returns
    -------
    aff_plus : 2D array
        Now P by Q where L = ``len(steps)`` and P == N+L, Q=N+L

    Examples
    --------
    >>> aff = np.eye(4)
    >>> aff[:3,:3] = np.arange(9).reshape((3,3))
    >>> append_diag(aff, [9, 10], [99,100])
    array([[  0.,   1.,   2.,   0.,   0.,   0.],
           [  3.,   4.,   5.,   0.,   0.,   0.],
           [  6.,   7.,   8.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   9.,   0.,  99.],
           [  0.,   0.,   0.,   0.,  10., 100.],
           [  0.,   0.,   0.,   0.,   0.,   1.]])
    """
    aff = np.asarray(aff)
    steps = np.atleast_1d(steps)
    starts = np.atleast_1d(starts)
    n_steps = len(steps)
    if len(starts) == 0:
        starts = np.zeros(n_steps, dtype=steps.dtype)
    elif len(starts) != n_steps:
        raise AffineError('Steps should have same length as starts')
    old_n_out, old_n_in = (aff.shape[0] - 1, aff.shape[1] - 1)
    aff_plus = np.zeros((old_n_out + n_steps + 1, old_n_in + n_steps + 1), dtype=aff.dtype)
    aff_plus[:old_n_out, :old_n_in] = aff[:old_n_out, :old_n_in]
    aff_plus[:old_n_out, -1] = aff[:old_n_out, -1]
    for i, el in enumerate(steps):
        aff_plus[old_n_out + i, old_n_in + i] = el
    aff_plus[old_n_out:, -1] = list(starts) + [1]
    return aff_plus