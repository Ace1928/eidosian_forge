from __future__ import annotations
import typing
import numpy as np
from .._utils import array_kind
def kde(data: FloatArray, grid: FloatArray, package: str, **kwargs: Any) -> FloatArray:
    """
    Kernel Density Estimation

    Parameters
    ----------
    package :
        Package whose kernel density estimation to use.
        Should be one of
        `['statsmodels-u', 'statsmodels-m', 'scipy', 'sklearn']`.
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
    if package == 'statsmodels':
        package = 'statsmodels-m'
    func = KDE_FUNCS[package]
    return func(data, grid, **kwargs)