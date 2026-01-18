from numpy.testing import assert_equal
import numpy as np
def groupmean_d(x, d):
    """groupmeans using dummy variables

    Parameters
    ----------
    x : array_like, ndim
        data array, tested for 1,2 and 3 dimensions
    d : ndarray, 1d
        dummy variable, needs to have the same length
        as x in axis 0.

    Returns
    -------
    groupmeans : ndarray, ndim-1
        means for each group along axis 0, the levels
        of the groups are the last axis

    Notes
    -----
    This will be memory intensive if there are many levels
    in the categorical variable, i.e. many columns in the
    dummy variable. In this case it is recommended to use
    a more efficient version.

    """
    x = np.asarray(x)
    nvars = x.ndim + 1
    sli = [slice(None)] + [None] * (nvars - 2) + [slice(None)]
    return (x[..., None] * d[sli]).sum(0) * 1.0 / d.sum(0)