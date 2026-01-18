import numpy as np
from numpy.testing import (
import pytest
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.util import seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.vecm import (
def load_results_statsmodels_exog(dataset):
    """
    Load data with seasonal terms in `exog`.

    Same as load_results_statsmodels() except that the seasonal term is
    provided to :class:`VECM`'s `__init__()` method via the `eoxg` parameter.
    This is to check whether the same results are produced no matter whether
    `exog` or `seasons` is being used.

    Parameters
    ----------
    dataset : DataSet
    """
    results_per_deterministic_terms = dict.fromkeys(dataset.dt_s_list)
    endog = data[dataset]
    for dt_s_tup in dataset.dt_s_list:
        det_string = dt_s_tup[0]
        seasons = dt_s_tup[1]
        first_season = dt_s_tup[2]
        if seasons == 0:
            exog = None
        else:
            exog = seasonal_dummies(seasons, len(data[dataset]), first_season, centered=True)
            if 'lo' in dt_s_tup[0]:
                exog = np.hstack((exog, 1 + np.arange(len(endog)).reshape(-1, 1)))
                det_string = det_string[:-2]
        model = VECM(endog, exog, k_ar_diff=3, coint_rank=coint_rank, deterministic=det_string)
        results_per_deterministic_terms[dt_s_tup] = model.fit(method='ml')
    return results_per_deterministic_terms