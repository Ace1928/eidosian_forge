from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def get_cell_set(data, starts_with=None, ends_with=None, exact_word=None, regex=None):
    """Get a list of cells from data.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features] or [n_samples]
        Input pd.DataFrame, or list of cell names
    starts_with : str, list-like or None, optional (default: None)
        If not None, only return cell names that start with this prefix.
    ends_with : str, list-like or None, optional (default: None)
        If not None, only return cell names that end with this suffix.
    exact_word : str, list-like or None, optional (default: None)
        If not None, only return cell names that contain this exact word.
    regex : str, list-like or None, optional (default: None)
        If not None, only return cell names that match this regular expression.

    Returns
    -------
    cells : list-like, shape<=[n_features]
        List of matching cells
    """
    if not _is_1d(data):
        try:
            data = data.index.to_numpy()
        except AttributeError:
            raise TypeError('data must be a list of cell names or a pandas DataFrame. Got {}'.format(type(data).__name__))
    if starts_with is None and ends_with is None and (regex is None) and (exact_word is None):
        warnings.warn('No selection conditions provided. Returning all cells.', UserWarning)
    return _get_string_subset(data, starts_with=starts_with, ends_with=ends_with, exact_word=exact_word, regex=regex)