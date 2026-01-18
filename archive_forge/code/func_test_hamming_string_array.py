import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def test_hamming_string_array():
    a = np.array(['eggs', 'spam', 'spam', 'eggs', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'eggs', 'eggs', 'spam', 'eggs', 'eggs', 'eggs', 'eggs', 'eggs', 'spam'], dtype='|S4')
    b = np.array(['eggs', 'spam', 'spam', 'eggs', 'eggs', 'spam', 'spam', 'spam', 'spam', 'eggs', 'spam', 'eggs', 'spam', 'eggs', 'spam', 'spam', 'eggs', 'spam', 'spam', 'eggs'], dtype='|S4')
    desired = 0.45
    assert_allclose(whamming(a, b), desired)