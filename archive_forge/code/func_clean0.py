import numpy as np
import pandas as pd
import scipy.linalg
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import array_like
def clean0(matrix):
    """
    Erase columns of zeros: can save some time in pseudoinverse.

    Parameters
    ----------
    matrix : ndarray
        The array to clean.

    Returns
    -------
    ndarray
        The cleaned array.
    """
    colsum = np.add.reduce(matrix ** 2, 0)
    val = [matrix[:, i] for i in np.flatnonzero(colsum)]
    return np.array(np.transpose(val))