import numpy as np
from numpy.testing import assert_, assert_allclose, assert_raises
import statsmodels.datasets.macrodata.data as macro
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.var_model import VAR
from .JMulTi_results.parse_jmulti_var_output import (
def generate_exog_from_season(seasons, endog_len):
    """
    Translate seasons to exog matrix.

    Parameters
    ----------
    seasons : int
        Number of seasons.
    endog_len : int
        Number of observations.

    Returns
    -------
    exog : ndarray or None
        If seasonal deterministic terms exist, the corresponding exog-matrix is
        returned.
        Otherwise, None is returned.
    """
    exog_stack = []
    if seasons > 0:
        season_exog = np.zeros((seasons - 1, endog_len))
        for i in range(seasons - 1):
            season_exog[i, i::seasons] = 1
        season_exog = season_exog.T
        exog_stack.append(season_exog)
    if exog_stack != []:
        exog = np.column_stack(exog_stack)
    else:
        exog = None
    return exog