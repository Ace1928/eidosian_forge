from __future__ import annotations
import typing
import numpy as np
from .._utils import array_kind
def kde_scipy(data: FloatArray, grid: FloatArray, **kwargs: Any) -> FloatArray:
    """
    Kernel Density Estimation with Scipy

    Parameters
    ----------
    data :
        Data points used to compute a density estimator. It
        has `n x p` dimensions, representing n points and p
        variables.
    grid :
        Data points at which the desity will be estimated. It
        has `m x p` dimensions, representing m points and p
        variables.

    Returns
    -------
    out : numpy.array
        Density estimate. Has `m x 1` dimensions
    """
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data.T, **kwargs)
    return kde.evaluate(grid.T)