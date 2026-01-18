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
def distanz(x, y=None):
    """
    Calculate the distance between two colon vectors.

    Parameters
    ----------
    x : ndarray
        First colon vector
    y : ndarray
        Second colon vector

    Returns
    -------
    d : ndarray
        Distance between x and y

    Examples
    --------
    >>> from pygsp import utils
    >>> x = np.arange(3)
    >>> utils.distanz(x, x)
    array([[ 0.,  1.,  2.],
           [ 1.,  0.,  1.],
           [ 2.,  1.,  0.]])

    """
    try:
        x.shape[1]
    except IndexError:
        x = x.reshape(1, x.shape[0])
    if y is None:
        y = x
    else:
        try:
            y.shape[1]
        except IndexError:
            y = y.reshape(1, y.shape[0])
    rx, cx = x.shape
    ry, cy = y.shape
    if rx != ry:
        raise ValueError('The sizes of x and y do not fit')
    xx = (x * x).sum(axis=0)
    yy = (y * y).sum(axis=0)
    xy = np.dot(x.T, y)
    d = abs(np.kron(np.ones((cy, 1)), xx).T + np.kron(np.ones((cx, 1)), yy) - 2 * xy)
    return np.sqrt(d)