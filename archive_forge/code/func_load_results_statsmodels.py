import numpy as np
from numpy.testing import (
import pytest
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.util import seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.vecm import (
def load_results_statsmodels(dataset):
    results_per_deterministic_terms = dict.fromkeys(dataset.dt_s_list)
    for dt_s_tup in dataset.dt_s_list:
        model = VECM(data[dataset], k_ar_diff=3, coint_rank=coint_rank, deterministic=dt_s_tup[0], seasons=dt_s_tup[1], first_season=dt_s_tup[2])
        results_per_deterministic_terms[dt_s_tup] = model.fit(method='ml')
    return results_per_deterministic_terms