import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_argument_validation(self):
    assert_raises(ValueError, random.multivariate_hypergeometric, 10, 4)
    assert_raises(ValueError, random.multivariate_hypergeometric, [2, 3, 4], -1)
    assert_raises(ValueError, random.multivariate_hypergeometric, [-1, 2, 3], 2)
    assert_raises(ValueError, random.multivariate_hypergeometric, [2, 3, 4], 10)
    assert_raises(ValueError, random.multivariate_hypergeometric, [], 1)
    assert_raises(ValueError, random.multivariate_hypergeometric, [999999999, 101], 5, 1, 'marginals')
    int64_info = np.iinfo(np.int64)
    max_int64 = int64_info.max
    max_int64_index = max_int64 // int64_info.dtype.itemsize
    assert_raises(ValueError, random.multivariate_hypergeometric, [max_int64_index - 100, 101], 5, 1, 'count')