from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import optimize
from scipy.special import factorial, xlogy
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
from sklearn.metrics._regression import _check_reg_targets
from sklearn.model_selection import GridSearchCV
from sklearn.utils._testing import (
def test__check_reg_targets_exception():
    invalid_multioutput = 'this_value_is_not_valid'
    expected_message = "Allowed 'multioutput' string values are.+You provided multioutput={!r}".format(invalid_multioutput)
    with pytest.raises(ValueError, match=expected_message):
        _check_reg_targets([1, 2, 3], [[1], [2], [3]], invalid_multioutput)