from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def select_cols(data, *extra_data, idx=None, starts_with=None, ends_with=None, exact_word=None, regex=None):
    """Select columns from a data matrix.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[any, n_features], optional
        Optional additional data objects from which to select the same rows
    idx : list-like, shape=[m_features]
        Integer indices or string column names to be selected
    starts_with : str, list-like or None, optional (default: None)
        If not None, select columns that start with this prefix.
    ends_with : str, list-like or None, optional (default: None)
        If not None, select columns that end with this suffix.
    exact_word : str, list-like or None, optional (default: None)
        If not None, select columns that contain this exact word.
    regex : str, list-like or None, optional (default: None)
        If not None, select columns that match this regular expression.

    Returns
    -------
    data : array-like, shape=[n_samples, m_features]
        Subsetted output data.
    extra_data : array-like, shape=[any, m_features]
        Subsetted extra data, if passed.

    Examples
    --------
    data_subset = scprep.select.select_cols(
        data,
        idx=np.random.choice([True, False],
        data.shape[1])
    )
    data_subset, metadata_subset = scprep.select.select_cols(
        data,
        metadata,
        starts_with="MT"
    )

    Raises
    ------
    UserWarning : if no columns are selected
    """
    if len(extra_data) > 0:
        _check_columns_compatible(data, *extra_data)
    if idx is None and starts_with is None and (ends_with is None) and (exact_word is None) and (regex is None):
        warnings.warn('No selection conditions provided. Returning all columns.', UserWarning)
        return tuple([data] + list(extra_data)) if len(extra_data) > 0 else data
    if idx is None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError('Can only select based on column names with DataFrame input. Please set `idx` to select specific columns.')
        idx = get_gene_set(data, starts_with=starts_with, ends_with=ends_with, exact_word=exact_word, regex=regex)
    if isinstance(idx, pd.DataFrame):
        idx = _convert_dataframe_1d(idx)
    elif not isinstance(idx, (numbers.Integral, str)):
        idx = utils.toarray(idx)
        _check_idx_1d(idx)
        idx = idx.flatten()
    if utils.is_SparseDataFrame(data):
        data = utils.SparseDataFrame(data)
    input_1d = _is_1d(data)
    if isinstance(data, pd.DataFrame):
        try:
            if isinstance(idx, (numbers.Integral, str)):
                data = data.loc[:, idx]
            else:
                if np.issubdtype(idx.dtype, np.dtype(bool).type):
                    raise TypeError
                data = data.loc[:, idx]
        except (KeyError, TypeError):
            if isinstance(idx, str):
                raise
            if isinstance(idx, numbers.Integral) or np.issubdtype(idx.dtype, np.dtype(int)) or np.issubdtype(idx.dtype, np.dtype(bool)):
                data = data.loc[:, np.array(data.columns)[idx]]
            else:
                raise
    elif isinstance(data, pd.Series):
        try:
            if np.issubdtype(idx.dtype, np.dtype(bool).type):
                raise TypeError
            data = data.loc[idx]
        except (KeyError, TypeError):
            if isinstance(idx, numbers.Integral) or np.issubdtype(idx.dtype, np.dtype(int)) or np.issubdtype(idx.dtype, np.dtype(bool)):
                data = data.loc[np.array(data.index)[idx]]
            else:
                raise
    elif _is_1d(data):
        if isinstance(data, list):
            data = np.array(data)
        data = data[idx]
    else:
        if isinstance(data, (sparse.coo_matrix, sparse.bsr_matrix, sparse.lil_matrix, sparse.dia_matrix)):
            data = data.tocsr()
        if isinstance(idx, pd.Series):
            idx = utils.toarray(idx)
        data = data[:, idx]
    if _get_column_length(data) == 0:
        warnings.warn('Selecting 0 columns.', UserWarning)
    elif isinstance(data, pd.DataFrame) and (not input_1d):
        data = _convert_dataframe_1d(data, silent=True)
    if len(extra_data) > 0:
        data = [data]
        for d in extra_data:
            data.append(select_cols(d, idx=idx))
        data = tuple(data)
    return data