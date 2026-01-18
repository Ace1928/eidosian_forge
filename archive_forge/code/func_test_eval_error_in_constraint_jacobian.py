import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP, AmplNLP
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import tempfile
from pyomo.contrib.pynumero.interfaces.utils import (
def test_eval_error_in_constraint_jacobian(self):
    m = self._make_bad_model()
    m.x[2] = -1
    nlp = PyomoNLP(m)
    msg = 'Error in AMPL evaluation'
    with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
        jacobian = nlp.evaluate_jacobian()