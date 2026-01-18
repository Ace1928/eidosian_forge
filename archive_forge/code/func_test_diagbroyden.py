from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def test_diagbroyden(self):
    x = nonlin.diagbroyden(F, F.xin, iter=11, alpha=1)
    assert_(nonlin.norm(x) < 1e-08)
    assert_(nonlin.norm(F(x)) < 1e-08)