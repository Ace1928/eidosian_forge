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
def engine(self, scramble: bool, seed=170382760648021597650530316304495310428, **kwargs) -> QMCEngine:
    if self.can_scramble:
        return self.qmce(scramble=scramble, seed=seed, **kwargs)
    elif scramble:
        pytest.skip()
    else:
        return self.qmce(seed=seed, **kwargs)