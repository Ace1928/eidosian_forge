import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.genmod.families import family
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.tools import add_constant
def test_glm_attr(self):
    for attr in ['llf', 'null_deviance', 'aic', 'df_resid', 'df_model', 'pearson_chi2', 'scale']:
        assert_allclose(getattr(self.res1, attr), getattr(self.res2, attr), rtol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        assert_allclose(self.res1.bic, self.res2.bic, rtol=1e-10)