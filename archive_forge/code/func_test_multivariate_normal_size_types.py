import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
def test_multivariate_normal_size_types(self):
    random.multivariate_normal([0], [[0]], size=1)
    random.multivariate_normal([0], [[0]], size=np.int_(1))
    random.multivariate_normal([0], [[0]], size=np.int64(1))