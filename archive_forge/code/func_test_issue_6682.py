import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_issue_6682():
    assert_allclose(nbinom.sf(250, 50, 32.0 / 63.0), 1.460458510976452e-35)