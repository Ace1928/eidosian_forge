import os
import numpy as np
from numpy.testing import (
def test_ndenumerate_crash(self):
    list(np.ndenumerate(np.array([[]])))