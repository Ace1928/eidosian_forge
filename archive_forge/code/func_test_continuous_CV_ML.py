import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_continuous_CV_ML(self):
    dens_ml = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp], exog=[self.growth], dep_type='c', indep_type='c', bw='cv_ml')
    npt.assert_allclose(dens_ml.bw, [0.5341164, 0.04510836], atol=0.001)