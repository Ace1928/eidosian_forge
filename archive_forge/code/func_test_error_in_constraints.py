import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
@unittest.skipUnless(cyipopt_ge_1_3, 'cyipopt version < 1.3.0')
def test_error_in_constraints(self):
    m, nlp, interface, bad_x = _get_model_nlp_interface(halt_on_evaluation_error=False)
    msg = 'Error in constraint evaluation'
    with self.assertRaisesRegex(cyipopt.CyIpoptEvaluationError, msg):
        interface.constraints(bad_x)