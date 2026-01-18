import itertools
from pyomo.common.dependencies import (
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.common.dependencies.scipy import stats
imports_available = (
def fit_rect_dist(theta_values, alpha):
    """
    Fit an alpha-level rectangular distribution to theta values

    Parameters
    ----------
    theta_values: DataFrame
        Theta values, columns = variable names
    alpha: float, optional
        Confidence interval value

    Returns
    ---------
    tuple containing lower bound and upper bound for each variable
    """
    assert isinstance(theta_values, pd.DataFrame)
    assert isinstance(alpha, (int, float))
    tval = stats.t.ppf(1 - (1 - alpha) / 2, len(theta_values) - 1)
    m = theta_values.mean()
    s = theta_values.std()
    lower_bound = m - tval * s
    upper_bound = m + tval * s
    return (lower_bound, upper_bound)