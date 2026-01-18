import numpy as np
from numpy.testing import assert_allclose
import pytest
from statsmodels.regression.linear_model import WLS
from statsmodels.regression._tools import _MinimalWLS
@pytest.mark.parametrize('bad_value', [np.nan, np.inf])
def test_inf_nan(self, bad_value):
    with pytest.raises(ValueError, match='detected in endog, estimation infeasible'):
        endog = self.endog1.copy()
        endog[0] = bad_value
        _MinimalWLS(endog, self.exog1, self.weights1, check_endog=True, check_weights=True).fit()
    with pytest.raises(ValueError, match='detected in weights, estimation infeasible'):
        weights = self.weights1.copy()
        weights[-1] = bad_value
        _MinimalWLS(self.endog1, self.exog1, weights, check_endog=True, check_weights=True).fit()