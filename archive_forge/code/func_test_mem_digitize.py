import os
import numpy as np
from numpy.testing import (
def test_mem_digitize(self):
    for i in range(100):
        np.digitize([1, 2, 3, 4], [1, 3])
        np.digitize([0, 1, 2, 3, 4], [1, 3])