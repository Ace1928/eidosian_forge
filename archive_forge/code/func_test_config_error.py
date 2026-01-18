import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
@unittest.skipIf(cyipopt_ge_1_3, 'cyipopt version >= 1.3.0')
def test_config_error(self):
    _, nlp, _, _ = _get_model_nlp_interface()
    with self.assertRaisesRegex(ValueError, 'halt_on_evaluation_error'):
        interface = CyIpoptNLP(nlp, halt_on_evaluation_error=False)