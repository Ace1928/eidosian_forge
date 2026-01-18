import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
@pytest.mark.parametrize('test', BOOST_TESTS, ids=repr)
def test_boost(test):
    if test.func in [btdtr, btdtri, btdtri_comp]:
        with pytest.deprecated_call():
            _test_factory(test)
    else:
        _test_factory(test)