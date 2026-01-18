from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def get_gene_set(data, starts_with=None, ends_with=None, exact_word=None, regex=None):
    """Get a list of genes from data.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features] or [n_features]
        Input pd.DataFrame, or list of gene names
    starts_with : str, list-like or None, optional (default: None)
        If not None, only return gene names that start with this prefix.
    ends_with : str, list-like or None, optional (default: None)
        If not None, only return gene names that end with this suffix.
    exact_word : str, list-like or None, optional (default: None)
        If not None, only return gene names that contain this exact word.
    regex : str, list-like or None, optional (default: None)
        If not None, only return gene names that match this regular expression.

    Returns
    -------
    genes : list-like, shape<=[n_features]
        List of matching genes
    """
    if not _is_1d(data):
        try:
            data = data.columns.to_numpy()
        except AttributeError:
            raise TypeError('data must be a list of gene names or a pandas DataFrame. Got {}'.format(type(data).__name__))
    if starts_with is None and ends_with is None and (regex is None) and (exact_word is None):
        warnings.warn('No selection conditions provided. Returning all genes.', UserWarning)
    return _get_string_subset(data, starts_with=starts_with, ends_with=ends_with, exact_word=exact_word, regex=regex)