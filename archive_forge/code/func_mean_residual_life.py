import numpy as np
def mean_residual_life(x, frac=None, alpha=0.05):
    """empirical mean residual life or expected shortfall

    Parameters
    ----------
    x : 1-dimensional array_like
    frac : list[float], optional
        All entries must be between 0 and 1
    alpha : float, default 0.05
        FIXME: not actually used.

    TODO:
        check formula for std of mean
        does not include case for all observations
        last observations std is zero
        vectorize loop using cumsum
        frac does not work yet
    """
    axis = 0
    x = np.asarray(x)
    nobs = x.shape[axis]
    xsorted = np.sort(x, axis=axis)
    if frac is None:
        xthreshold = xsorted
    else:
        xthreshold = xsorted[np.floor(nobs * frac).astype(int)]
    xlargerindex = np.searchsorted(xsorted, xthreshold, side='right')
    result = []
    for i in range(len(xthreshold) - 1):
        k_ind = xlargerindex[i]
        rmean = x[k_ind:].mean()
        rstd = x[k_ind:].std()
        rmstd = rstd / np.sqrt(nobs - k_ind)
        result.append((k_ind, xthreshold[i], rmean, rmstd))
    res = np.array(result)
    crit = 1.96
    confint = res[:, 1:2] + crit * res[:, -1:] * np.array([[-1, 1]])
    return np.column_stack((res, confint))