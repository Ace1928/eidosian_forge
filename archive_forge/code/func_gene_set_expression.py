from . import select
from . import utils
from scipy import sparse
import numpy as np
import pandas as pd
import scipy.signal
def gene_set_expression(data, genes=None, library_size_normalize=False, starts_with=None, ends_with=None, exact_word=None, regex=None):
    """Measure the expression of a set of genes in each cell.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    genes : list-like, shape<=[n_features], optional (default: None)
        Integer column indices or string gene names included in gene set
    library_size_normalize : bool, optional (default: False)
        Divide gene set expression by library size
    starts_with : str or None, optional (default: None)
        If not None, select genes that start with this prefix
    ends_with : str or None, optional (default: None)
        If not None, select genes that end with this suffix
    exact_word : str, list-like or None, optional (default: None)
        If not None, select genes that contain this exact word.
    regex : str or None, optional (default: None)
        If not None, select genes that match this regular expression

    Returns
    -------
    gene_set_expression : list-like, shape=[n_samples]
        Sum over genes for each cell
    """
    if library_size_normalize:
        from .normalize import library_size_normalize
        data = library_size_normalize(data)
    gene_data = select.select_cols(data, idx=genes, starts_with=starts_with, ends_with=ends_with, exact_word=exact_word, regex=regex)
    if len(gene_data.shape) > 1:
        gene_set_expression = library_size(gene_data)
    else:
        gene_set_expression = gene_data
    if isinstance(gene_set_expression, pd.Series):
        gene_set_expression.name = 'expression'
    return gene_set_expression