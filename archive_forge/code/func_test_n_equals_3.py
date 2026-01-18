import itertools
import sys
import pytest
import numpy as np
from numpy.testing import assert_
from scipy.special._testutils import FuncData
from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi
from scipy.special._ufuncs import (_kolmogc, _kolmogci, _kolmogp,
def test_n_equals_3(self):
    x = np.linspace(0.7, 1, 31, endpoint=True)
    p = np.power(1 - x, 3)
    n = np.array([3] * len(x))
    dataset = np.column_stack([n, p, x])
    FuncData(smirnovi, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
    dataset[:, 1] = 1 - dataset[:, 1]
    FuncData(_smirnovci, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])