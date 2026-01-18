from k-means models and quantizing vectors by comparing them with
import warnings
import numpy as np
from collections import deque
from scipy._lib._array_api import (
from scipy._lib._util import check_random_state, rng_integers
from scipy.spatial.distance import cdist
from . import _vq
def py_vq(obs, code_book, check_finite=True):
    """ Python version of vq algorithm.

    The algorithm computes the Euclidean distance between each
    observation and every frame in the code_book.

    Parameters
    ----------
    obs : ndarray
        Expects a rank 2 array. Each row is one observation.
    code_book : ndarray
        Code book to use. Same format than obs. Should have same number of
        features (e.g., columns) than obs.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True

    Returns
    -------
    code : ndarray
        code[i] gives the label of the ith obversation; its code is
        code_book[code[i]].
    mind_dist : ndarray
        min_dist[i] gives the distance between the ith observation and its
        corresponding code.

    Notes
    -----
    This function is slower than the C version but works for
    all input types. If the inputs have the wrong types for the
    C versions of the function, this one is called as a last resort.

    It is about 20 times slower than the C version.

    """
    xp = array_namespace(obs, code_book)
    obs = as_xparray(obs, xp=xp, check_finite=check_finite)
    code_book = as_xparray(code_book, xp=xp, check_finite=check_finite)
    if obs.ndim != code_book.ndim:
        raise ValueError('Observation and code_book should have the same rank')
    if obs.ndim == 1:
        obs = obs[:, xp.newaxis]
        code_book = code_book[:, xp.newaxis]
    dist = xp.asarray(cdist(obs, code_book))
    code = xp.argmin(dist, axis=1)
    min_dist = xp.min(dist, axis=1)
    return (code, min_dist)