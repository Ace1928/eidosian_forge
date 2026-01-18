import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from statsmodels.datasets import elnino
from statsmodels.graphics.functional import (
def harmfunc(t):
    """Test function, combination of a few harmonic terms."""
    ci = int(np.random.random() > 0.9)
    a1i = np.random.random() * 0.05
    a2i = np.random.random() * 0.05
    b1i = (0.15 - 0.1) * np.random.random() + 0.1
    b2i = (0.15 - 0.1) * np.random.random() + 0.1
    func = (1 - ci) * (a1i * np.sin(t) + a2i * np.cos(t)) + ci * (b1i * np.sin(t) + b2i * np.cos(t))
    return func