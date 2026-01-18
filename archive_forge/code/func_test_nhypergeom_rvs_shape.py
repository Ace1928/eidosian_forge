import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_nhypergeom_rvs_shape():
    x = nhypergeom.rvs(22, [7, 8, 9], [[12], [13]], size=(5, 1, 2, 3))
    assert x.shape == (5, 1, 2, 3)