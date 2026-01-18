from __future__ import annotations
import typing
import numpy as np
from .._utils import array_kind
def kde_statsmodels_u(data: FloatArray, grid: FloatArray, **kwargs: Any) -> FloatArray:
    """
    Univariate Kernel Density Estimation with Statsmodels

    Parameters
    ----------
    data :
        Data points used to compute a density estimator. It
        has `n x 1` dimensions, representing n points and p
        variables.
    grid :
        Data points at which the desity will be estimated. It
        has `m x 1` dimensions, representing m points and p
        variables.

    Returns
    -------
    out : numpy.array
        Density estimate. Has `m x 1` dimensions
    """
    from statsmodels.nonparametric.kde import KDEUnivariate
    kde = KDEUnivariate(data)
    kde.fit(**kwargs)
    return kde.evaluate(grid)