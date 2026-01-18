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
def disc_md(x):
    n, s = x.shape
    xij = x
    disc1 = np.sum(np.prod(5 / 3 - 1 / 4 * np.abs(xij - 0.5) - 1 / 4 * np.abs(xij - 0.5) ** 2, axis=1))
    xij = x[None, :, :]
    xkj = x[:, None, :]
    disc2 = np.sum(np.sum(np.prod(15 / 8 - 1 / 4 * np.abs(xij - 0.5) - 1 / 4 * np.abs(xkj - 0.5) - 3 / 4 * np.abs(xij - xkj) + 1 / 2 * np.abs(xij - xkj) ** 2, axis=2), axis=0))
    return (19 / 12) ** s - 2 / n * disc1 + 1 / n ** 2 * disc2