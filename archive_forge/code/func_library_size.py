from . import select
from . import utils
from scipy import sparse
import numpy as np
import pandas as pd
import scipy.signal
def library_size(data):
    """Measure the library size of each cell.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data

    Returns
    -------
    library_size : list-like, shape=[n_samples]
        Sum over all genes for each cell
    """
    library_size = utils.matrix_sum(data, axis=1)
    if isinstance(library_size, pd.Series):
        library_size.name = 'library_size'
    return library_size