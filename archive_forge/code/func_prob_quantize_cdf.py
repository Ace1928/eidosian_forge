from statsmodels.compat.python import lmap
import numpy as np
def prob_quantize_cdf(binsx, binsy, cdf):
    """quantize a continuous distribution given by a cdf

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
    cdf_values = cdf(binsx[:, None], binsy)
    cdf_func = lambda x, y: cdf_values[x, y]
    for xind in range(1, nx + 1):
        for yind in range(1, ny + 1):
            upper = (xind, yind)
            lower = (xind - 1, yind - 1)
            probs[xind - 1, yind - 1] = prob_bv_rectangle(lower, upper, cdf_func)
    assert not np.isnan(probs).any()
    return probs