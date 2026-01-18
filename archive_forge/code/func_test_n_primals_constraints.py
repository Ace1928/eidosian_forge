import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
def test_n_primals_constraints(self):
    m, nlp, proj_nlp = self._get_nlps()
    self.assertEqual(proj_nlp.n_primals(), 2)
    self.assertEqual(proj_nlp.n_constraints(), 5)
    self.assertEqual(proj_nlp.n_eq_constraints(), 2)
    self.assertEqual(proj_nlp.n_ineq_constraints(), 3)