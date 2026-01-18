from statsmodels.compat.python import lzip, lmap
from numpy.testing import (
import numpy as np
import pytest
from statsmodels.stats.libqsturng import qsturng, psturng
def test_invalid_parameters(self):
    assert_raises(ValueError, psturng, -0.1, 5, 6)
    assert_raises((ValueError, OverflowError), psturng, 0.9, 1, 2)