from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def check_consistent_columns(data, common_columns_only=True):
    """Ensure that a set of data matrices have consistent columns.

    Parameters
    ----------
    data : list of array-likes
        List of matrices to be checked
    common_columns_only : bool, optional (default: True)
        With pandas inputs, drop any columns that are not common to
        all matrices

    Returns
    -------
    data : list of array-likes
        List of matrices with consistent columns, subsetted if necessary

    Raises
    ------
    ValueError
        Raised if data has inconsistent number of columns and does not
        have column names for subsetting
    """
    matrix_type = type(data[0])
    matrix_shape = data[0].shape[1]
    if issubclass(matrix_type, pd.DataFrame):
        if not (np.all([d.shape[1] == matrix_shape for d in data[1:]]) and np.all([data[0].columns == d.columns for d in data])):
            if common_columns_only:
                common_genes = data[0].columns.values
                for d in data[1:]:
                    common_genes = common_genes[np.isin(common_genes, d.columns.values)]
                warnings.warn('Input data has inconsistent column names. Subsetting to {} common columns. To retain all columns, use `common_columns_only=False`.'.format(len(common_genes)), UserWarning)
                for i in range(len(data)):
                    data[i] = data[i][common_genes]
            else:
                columns = [d.columns.values for d in data]
                all_columns = np.unique(np.concatenate(columns))
                warnings.warn('Input data has inconsistent column names. Padding with zeros to {} total columns.'.format(len(all_columns)), UserWarning)
    else:
        for d in data[1:]:
            if not d.shape[1] == matrix_shape:
                shapes = ', '.join([str(d.shape[1]) for d in data])
                raise ValueError('Expected data all with the same number of columns. Got {}'.format(shapes))
    return data