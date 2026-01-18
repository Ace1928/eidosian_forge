from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def sparse_series_min(data):
    """Get the minimum value from a pandas sparse series.

    Pandas SparseDataFrame does not handle np.min.

    Parameters
    ----------
    data : pd.Series[SparseArray]
        Input data

    Returns
    -------
    minimum : float
        Minimum entry in `data`.
    """
    return np.concatenate([data.sparse.sp_values, [data.sparse.fill_value]]).min()