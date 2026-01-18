import os
from collections import Counter
from itertools import combinations, product
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from scipy.spatial import distance
from scipy.stats import shapiro
from scipy.stats._sobol import _test_find_index
from scipy.stats import qmc
from scipy.stats._qmc import (
def test_n_primes(self):
    primes = n_primes(10)
    assert primes[-1] == 29
    primes = n_primes(168)
    assert primes[-1] == 997
    primes = n_primes(350)
    assert primes[-1] == 2357