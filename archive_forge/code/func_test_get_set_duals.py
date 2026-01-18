import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
def test_get_set_duals(self):
    m, nlp, proj_nlp = self._get_nlps()
    nlp.set_duals([2, 3, 4, 5, 6])
    np.testing.assert_array_equal(proj_nlp.get_duals(), [2, 3, 4, 5, 6])
    proj_nlp.set_duals([-1, -2, -3, -4, -5])
    np.testing.assert_array_equal(proj_nlp.get_duals(), [-1, -2, -3, -4, -5])
    np.testing.assert_array_equal(nlp.get_duals(), [-1, -2, -3, -4, -5])