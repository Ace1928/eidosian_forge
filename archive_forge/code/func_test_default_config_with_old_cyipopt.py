import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
@unittest.skipIf(cyipopt_ge_1_3, 'cyipopt version >= 1.3.0')
def test_default_config_with_old_cyipopt(self):
    _, nlp, _, bad_x = _get_model_nlp_interface()
    interface = CyIpoptNLP(nlp)
    msg = 'Error in AMPL evaluation'
    with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
        interface.objective(bad_x)