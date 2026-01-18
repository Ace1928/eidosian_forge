import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
@pytest.mark.slow
def test_mixeddata_CV_LS(self):
    dens_ls = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp], exog=[self.Italy_year], dep_type='c', indep_type='o', bw='cv_ls')
    npt.assert_allclose(dens_ls.bw, [1.01203728, 0.31905144], atol=1e-05)