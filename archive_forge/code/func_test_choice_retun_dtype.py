import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
def test_choice_retun_dtype(self):
    c = np.random.choice(10, p=[0.1] * 10, size=2)
    assert c.dtype == np.dtype(int)
    c = np.random.choice(10, p=[0.1] * 10, replace=False, size=2)
    assert c.dtype == np.dtype(int)
    c = np.random.choice(10, size=2)
    assert c.dtype == np.dtype(int)
    c = np.random.choice(10, replace=False, size=2)
    assert c.dtype == np.dtype(int)