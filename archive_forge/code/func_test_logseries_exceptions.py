import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
@pytest.mark.parametrize('value', [np.nextafter(0.0, -1), 1.0, np.nan, 5.0])
def test_logseries_exceptions(self, value):
    with np.errstate(invalid='ignore'):
        with pytest.raises(ValueError):
            random.logseries(value)
        with pytest.raises(ValueError):
            random.logseries(np.array([value] * 10))
        with pytest.raises(ValueError):
            random.logseries(np.array([value] * 10)[::2])