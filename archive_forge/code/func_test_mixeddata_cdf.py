import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
@pytest.mark.slow
def test_mixeddata_cdf(self):
    dens = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp], exog=[self.Italy_year], dep_type='c', indep_type='o', bw='cv_ls')
    sm_result = dens.cdf()[0:5]
    expected = [0.83378885, 0.97684477, 0.90655143, 0.79393161, 0.43629083]
    npt.assert_allclose(sm_result, expected, atol=0, rtol=1e-05)