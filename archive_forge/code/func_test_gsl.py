import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
@pytest.mark.parametrize('test', GSL_TESTS, ids=repr)
def test_gsl(test):
    _test_factory(test)