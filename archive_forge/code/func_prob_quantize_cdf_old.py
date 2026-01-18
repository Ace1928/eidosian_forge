from statsmodels.compat.python import lmap
import numpy as np
def prob_quantize_cdf_old(binsx, binsy, cdf):
    """quantize a continuous distribution given by a cdf

    old version without precomputing cdf values

    Parameters
    ----------
    binsx : array_like, 1d
        binedges

    """
    binsx = np.asarray(binsx)
    binsy = np.asarray(binsy)
    nx = len(binsx) - 1
    ny = len(binsy) - 1
    probs = np.nan * np.ones((nx, ny))
    for xind in range(1, nx + 1):
        for yind in range(1, ny + 1):
            upper = (binsx[xind], binsy[yind])
            lower = (binsx[xind - 1], binsy[yind - 1])
            probs[xind - 1, yind - 1] = prob_bv_rectangle(lower, upper, cdf)
    assert not np.isnan(probs).any()
    return probs