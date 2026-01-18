import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
def dispersion_poisson(results):
    """Score/LM type tests for Poisson variance assumptions

    .. deprecated:: 0.14

       dispersion_poisson moved to discrete._diagnostic_count

    Null Hypothesis is

    H0: var(y) = E(y) and assuming E(y) is correctly specified
    H1: var(y) ~= E(y)

    The tests are based on the constrained model, i.e. the Poisson model.
    The tests differ in their assumed alternatives, and in their maintained
    assumptions.

    Parameters
    ----------
    results : Poisson results instance
        This can be a results instance for either a discrete Poisson or a GLM
        with family Poisson.

    Returns
    -------
    res : ndarray, shape (7, 2)
       each row contains the test statistic and p-value for one of the 7 tests
       computed here.
    description : 2-D list of strings
       Each test has two strings a descriptive name and a string for the
       alternative hypothesis.
    """
    msg = 'dispersion_poisson here is deprecated, use the version in discrete._diagnostic_count'
    warnings.warn(msg, FutureWarning)
    from statsmodels.discrete._diagnostics_count import test_poisson_dispersion
    return test_poisson_dispersion(results, _old=True)