import numpy as np
from numpy.testing import (
def test_mapping_error_message(self):
    a = np.zeros((3, 5))
    index = (1, 2, 3, 4, 5)
    assert_raises_regex(IndexError, 'too many indices for array: array is 2-dimensional, but 5 were indexed', lambda: a[index])