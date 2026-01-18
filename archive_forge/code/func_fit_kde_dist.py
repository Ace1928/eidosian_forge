import itertools
from pyomo.common.dependencies import (
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.common.dependencies.scipy import stats
imports_available = (
def fit_kde_dist(theta_values):
    """
    Fit a Gaussian kernel-density distribution to theta values

    Parameters
    ----------
    theta_values: DataFrame
        Theta values, columns = variable names

    Returns
    ---------
    scipy.stats.gaussian_kde distribution
    """
    assert isinstance(theta_values, pd.DataFrame)
    dist = stats.gaussian_kde(theta_values.transpose().values)
    return dist