import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_raises,
import numpy.testing as npt
from scipy.special import gamma, factorial, factorial2
import scipy.stats as stats
from statsmodels.distributions.edgeworth import (_faa_di_bruno_partitions,
def test_badvalues(self):
    assert_raises(ValueError, cumulant_from_moments, [1, 2, 3], 0)
    assert_raises(ValueError, cumulant_from_moments, [1, 2, 3], 4)