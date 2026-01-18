from __future__ import annotations
import typing
import numpy as np
from .._utils import array_kind
def kde_sklearn(data: FloatArray, grid: FloatArray, **kwargs: Any) -> FloatArray:
    """
    Kernel Density Estimation with Scikit-learn

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
    out :
        Density estimate. Has `m x 1` dimensions
    """
    try:
        from sklearn.neighbors import KernelDensity
    except ImportError as err:
        raise ImportError('scikit-learn is not installed') from err
    kde_skl = KernelDensity(**kwargs)
    kde_skl.fit(data)
    log_pdf = kde_skl.score_samples(grid)
    return np.exp(log_pdf)