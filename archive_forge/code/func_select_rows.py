from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def select_rows(data, *extra_data, idx=None, starts_with=None, ends_with=None, exact_word=None, regex=None):
    """Select rows from a data matrix.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[n_samples, any], optional
        Optional additional data objects from which to select the same rows
    idx : list-like, shape=[m_samples], optional (default: None)
        Integer indices or string index names to be selected
    starts_with : str, list-like or None, optional (default: None)
        If not None, select rows that start with this prefix.
    ends_with : str, list-like or None, optional (default: None)
        If not None, select rows that end with this suffix.
    exact_word : str, list-like or None, optional (default: None)
        If not None, select rows that contain this exact word.
    regex : str, list-like or None, optional (default: None)
        If not None, select rows that match this regular expression.

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Subsetted output data
    extra_data : array-like, shape=[m_samples, any]
        Subsetted extra data, if passed.

    Examples
    --------
    data_subset = scprep.select.select_rows(
        data,
        idx=np.random.choice([True, False],
        data.shape[0])
    )
    data_subset, labels_subset = scprep.select.select_rows(
        data,
        labels,
        end_with="batch1"
    )

    Raises
    ------
    UserWarning : if no rows are selected
    """
    if len(extra_data) > 0:
        _check_rows_compatible(data, *extra_data)
    if idx is None and starts_with is None and (ends_with is None) and (exact_word is None) and (regex is None):
        warnings.warn('No selection conditions provided. Returning all rows.', UserWarning)
        return tuple([data] + list(extra_data)) if len(extra_data) > 0 else data
    if idx is None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError('Can only select based on row names with DataFrame input. Please set `idx` to select specific rows.')
        idx = get_cell_set(data, starts_with=starts_with, ends_with=ends_with, exact_word=exact_word, regex=regex)
    if isinstance(idx, pd.DataFrame):
        idx = _convert_dataframe_1d(idx)
    elif not isinstance(idx, (numbers.Integral, str)):
        idx = utils.toarray(idx)
        _check_idx_1d(idx)
        idx = idx.flatten()
    if utils.is_SparseDataFrame(data):
        data = utils.SparseDataFrame(data)
    input_1d = _is_1d(data)
    if isinstance(data, (pd.DataFrame, pd.Series)):
        try:
            if isinstance(idx, (numbers.Integral, str)):
                data = data.loc[idx]
            else:
                if np.issubdtype(idx.dtype, np.dtype(bool).type):
                    raise TypeError
                with warnings.catch_warnings():
                    warnings.filterwarnings('error', 'Passing list-likes to .loc')
                    data = data.loc[idx]
        except (KeyError, TypeError, FutureWarning):
            if isinstance(idx, str):
                raise
            if isinstance(idx, numbers.Integral) or np.issubdtype(idx.dtype, np.dtype(int)) or np.issubdtype(idx.dtype, np.dtype(bool)):
                data = data.loc[np.array(data.index)[idx]]
            else:
                raise
    elif _is_1d(data):
        if isinstance(data, list):
            data = np.array(data)
        data = data[idx]
    else:
        if isinstance(data, (sparse.coo_matrix, sparse.bsr_matrix, sparse.dia_matrix)):
            data = data.tocsr()
        if isinstance(idx, pd.Series):
            idx = utils.toarray(idx)
        data = data[idx, :]
    if _get_row_length(data) == 0:
        warnings.warn('Selecting 0 rows.', UserWarning)
    elif isinstance(data, pd.DataFrame) and (not input_1d):
        data = _convert_dataframe_1d(data, silent=True)
    if len(extra_data) > 0:
        data = [data]
        for d in extra_data:
            data.append(select_rows(d, idx=idx))
        data = tuple(data)
    return data