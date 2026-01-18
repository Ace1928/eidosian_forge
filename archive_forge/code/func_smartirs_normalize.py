import logging
from functools import partial
import re
import numpy as np
from gensim import interfaces, matutils, utils
from gensim.utils import deprecated
@deprecated('Function will be removed in 4.0.0')
def smartirs_normalize(x, norm_scheme, return_norm=False):
    """Normalize a vector using the normalization scheme specified in `norm_scheme`.

    Parameters
    ----------
    x : numpy.ndarray
        The tf-idf vector.
    norm_scheme : {'n', 'c'}
        Document length normalization scheme.
    return_norm : bool, optional
        Return the length of `x` as well?

    Returns
    -------
    numpy.ndarray
        Normalized array.
    float (only if return_norm is set)
        Norm of `x`.
    """
    if norm_scheme == 'n':
        if return_norm:
            _, length = matutils.unitvec(x, return_norm=return_norm)
            return (x, length)
        else:
            return x
    elif norm_scheme == 'c':
        return matutils.unitvec(x, return_norm=return_norm)