from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
def scotts_factor(kde_obj):
    """Same as default, just check that it works."""
    return np.power(kde_obj.neff, -1.0 / (kde_obj.d + 4))