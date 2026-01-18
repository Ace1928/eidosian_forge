import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_3D_array(self):
    a = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]])
    res = dsplit(a, 2)
    desired = [np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]]), np.array([[[3, 4], [3, 4]], [[3, 4], [3, 4]]])]
    compare_results(res, desired)