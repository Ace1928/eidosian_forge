from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def test_anderson(self):
    x = nonlin.anderson(F, F.xin, iter=12, alpha=0.03, M=5)
    assert_(nonlin.norm(x) < 0.33)