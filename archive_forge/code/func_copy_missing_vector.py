import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def copy_missing_vector(a, b, missing, inplace=False, prefix=None):
    """
    Reorder the elements of a time-varying vector where all non-missing
    values are in the first elements of the vector.

    Parameters
    ----------
    a : array_like
        The vector from which to copy. Must have shape (n, nobs) or (n, 1).
    b : array_like
        The vector to copy to. Must have shape (n, nobs).
    missing : array_like of bool
        The vector of missing indices. Must have shape (n, nobs).
    inplace : bool, optional
        Whether or not to copy to b in-place. Default is False.
    prefix : {'s', 'd', 'c', 'z'}, optional
        The Fortran prefix of the vector. Default is to automatically detect
        the dtype. This parameter should only be used with caution.

    Returns
    -------
    copied_vector : array_like
        The vector b with the non-missing subvector of b copied onto it.
    """
    if prefix is None:
        prefix = find_best_blas_type((a, b))[0]
    copy = prefix_copy_missing_vector_map[prefix]
    if not inplace:
        b = np.copy(b, order='F')
    try:
        if not a.is_f_contig():
            raise ValueError()
    except (AttributeError, ValueError):
        a = np.asfortranarray(a)
    copy(a, b, np.asfortranarray(missing))
    return b