from statsmodels.compat.pandas import testing as pdt
import os.path
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.stats.outliers_influence import MLEInfluence
def test_influence_glm_bernoulli():
    df = data_bin
    results_sas = np.asarray(results_sas_df)
    res = GLM(df['constrict'], df[['const', 'log_rate', 'log_volumne']], family=families.Binomial()).fit(attach_wls=True, atol=1e-10)
    infl = res.get_influence(observed=False)
    k_vars = 3
    assert_allclose(infl.dfbetas, results_sas[:, 5:8], atol=0.0001)
    assert_allclose(infl.d_params, results_sas[:, 5:8] * res.bse.values, atol=0.0001)
    assert_allclose(infl.cooks_distance[0] * k_vars, results_sas[:, 8], atol=6e-05)
    assert_allclose(infl.hat_matrix_diag, results_sas[:, 4], atol=6e-05)
    c_bar = infl.cooks_distance[0] * 3 * (1 - infl.hat_matrix_diag)
    assert_allclose(c_bar, results_sas[:, 9], atol=6e-05)