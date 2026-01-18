import sys
import pytest
from numpy.testing import (
import numpy as np
from numpy import random
def test_logseries_convergence(self):
    N = 1000
    random.seed(0)
    rvsn = random.logseries(0.8, size=N)
    freq = np.sum(rvsn == 1) / N
    msg = f'Frequency was {freq:f}, should be > 0.45'
    assert_(freq > 0.45, msg)
    freq = np.sum(rvsn == 2) / N
    msg = f'Frequency was {freq:f}, should be < 0.23'
    assert_(freq < 0.23, msg)