from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def test_excitingmixing(self):
    self._check_dot(nonlin.ExcitingMixing, complex=False)
    self._check_dot(nonlin.ExcitingMixing, complex=True)