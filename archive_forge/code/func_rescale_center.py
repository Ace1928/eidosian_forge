from __future__ import division
import sys
import importlib
import logging
import functools
import pkgutil
import io
import numpy as np
from scipy import sparse
import scipy.io
def rescale_center(x):
    """
    Rescale and center data, e.g. embedding coordinates.

    Parameters
    ----------
    x : ndarray
        Data to be rescaled.

    Returns
    -------
    r : ndarray
        Rescaled data.

    Examples
    --------
    >>> from pygsp import utils
    >>> x = np.array([[1, 6], [2, 5], [3, 4]])
    >>> utils.rescale_center(x)
    array([[-1. ,  1. ],
           [-0.6,  0.6],
           [-0.2,  0.2]])

    """
    N = x.shape[1]
    y = x - np.kron(np.ones((1, N)), np.mean(x, axis=1)[:, np.newaxis])
    c = np.amax(y)
    r = y / c
    return r