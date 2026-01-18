import numpy as np
from numpy.testing import (
import pytest
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.util import seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.vecm import (
def load_results_statsmodels_exog_coint(dataset):
    """
    Load data with deterministic terms in `exog_coint`.

    Same as load_results_statsmodels() except that deterministic terms inside
    the cointegration relation are provided to :class:`VECM`'s `__init__()`
    method via the `eoxg_coint` parameter. This is to check whether the same
    results are produced no matter whether `exog_coint` or the `deterministic`
    argument is being used.

    Parameters
    ----------
    dataset : DataSet
    """
    results_per_deterministic_terms = dict.fromkeys(dataset.dt_s_list)
    endog = data[dataset]
    for dt_s_tup in dataset.dt_s_list:
        det_string = dt_s_tup[0]
        if 'ci' not in det_string and 'li' not in det_string:
            exog_coint = None
        else:
            exog_coint = []
            if 'li' in det_string:
                exog_coint.append(1 + np.arange(len(endog)).reshape(-1, 1))
                det_string = det_string[:-2]
            if 'ci' in det_string:
                exog_coint.append(np.ones(len(endog)).reshape(-1, 1))
                det_string = det_string[:-2]
            exog_coint = exog_coint[::-1]
            exog_coint = np.hstack(exog_coint)
        model = VECM(endog, exog=None, exog_coint=exog_coint, k_ar_diff=3, coint_rank=coint_rank, deterministic=det_string, seasons=dt_s_tup[1], first_season=dt_s_tup[2])
        results_per_deterministic_terms[dt_s_tup] = model.fit(method='ml')
    return results_per_deterministic_terms