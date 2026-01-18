from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def test_broyden2(self):
    x = nonlin.broyden2(F, F.xin, iter=12, alpha=1)
    assert_(nonlin.norm(x) < 1e-09)
    assert_(nonlin.norm(F(x)) < 1e-09)