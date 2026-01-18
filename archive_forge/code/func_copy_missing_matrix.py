import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def copy_missing_matrix(A, B, missing, missing_rows=False, missing_cols=False, is_diagonal=False, inplace=False, prefix=None):
    """
    Copy the rows or columns of a time-varying matrix where all non-missing
    values are in the upper left corner of the matrix.

    Parameters
    ----------
    A : array_like
        The matrix from which to copy. Must have shape (n, m, nobs) or
        (n, m, 1).
    B : array_like
        The matrix to copy to. Must have shape (n, m, nobs).
    missing : array_like of bool
        The vector of missing indices. Must have shape (k, nobs) where `k = n`
        if `reorder_rows is True` and `k = m` if `reorder_cols is True`.
    missing_rows : bool, optional
        Whether or not the rows of the matrix are a missing dimension. Default
        is False.
    missing_cols : bool, optional
        Whether or not the columns of the matrix are a missing dimension.
        Default is False.
    is_diagonal : bool, optional
        Whether or not the matrix is diagonal. If this is True, must also have
        `n = m`. Default is False.
    inplace : bool, optional
        Whether or not to copy to B in-place. Default is False.
    prefix : {'s', 'd', 'c', 'z'}, optional
        The Fortran prefix of the vector. Default is to automatically detect
        the dtype. This parameter should only be used with caution.

    Returns
    -------
    copied_matrix : array_like
        The matrix B with the non-missing submatrix of A copied onto it.
    """
    if prefix is None:
        prefix = find_best_blas_type((A, B))[0]
    copy = prefix_copy_missing_matrix_map[prefix]
    if not inplace:
        B = np.copy(B, order='F')
    try:
        if not A.is_f_contig():
            raise ValueError()
    except (AttributeError, ValueError):
        A = np.asfortranarray(A)
    copy(A, B, np.asfortranarray(missing), missing_rows, missing_cols, is_diagonal)
    return B