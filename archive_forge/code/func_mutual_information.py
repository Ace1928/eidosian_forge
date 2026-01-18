from . import plot
from . import select
from . import utils
from ._lazyload import matplotlib
from scipy import sparse
from scipy import stats
from sklearn import metrics
from sklearn import neighbors
import joblib
import numbers
import numpy as np
import pandas as pd
import warnings
def mutual_information(x, y, bins=8):
    """Compute mutual information score with set number of bins.

    Helper function for `sklearn.metrics.mutual_info_score` that builds a
    contingency table over a set number of bins.
    Credit: `Warran Weckesser <https://stackoverflow.com/a/20505476/3996580>`_.


    Parameters
    ----------
    x : array-like, shape=[n_samples]
        Input data (feature 1)
    y : array-like, shape=[n_samples]
        Input data (feature 2)
    bins : int or array-like, (default: 8)
        Passed to np.histogram2d to calculate a contingency table.

    Returns
    -------
    mi : float
        Mutual information between x and y.

    Examples
    --------
    >>> import scprep
    >>> data = scprep.io.load_csv("my_data.csv")
    >>> mi = scprep.stats.mutual_information(data['GENE1'], data['GENE2'])
    """
    x, y = _vector_coerce_two_dense(x, y)
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = metrics.mutual_info_score(None, None, contingency=c_xy)
    return mi