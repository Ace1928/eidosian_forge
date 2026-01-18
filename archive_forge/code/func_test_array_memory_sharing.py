import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_array_memory_sharing(self):
    assert_(np.may_share_memory(self.a, self.a.ravel()))
    assert_(not np.may_share_memory(self.a, self.a.flatten()))