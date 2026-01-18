import itertools
import sys
import pytest
import numpy as np
from numpy.testing import assert_
from scipy.special._testutils import FuncData
from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi
from scipy.special._ufuncs import (_kolmogc, _kolmogci, _kolmogp,
def test_n_large(self):
    x = 0.4
    pvals = np.array([smirnov(n, x) for n in range(400, 1100, 20)])
    dfs = np.diff(pvals)
    assert_(np.all(dfs <= 0), msg='Not all diffs negative %s' % dfs)