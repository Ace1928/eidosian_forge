import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose, assert_raises
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.arima import specification
def test_invalid_estimator():
    spec = specification.SARIMAXSpecification()
    assert_raises(ValueError, spec.validate_estimator, 'not_an_estimator')