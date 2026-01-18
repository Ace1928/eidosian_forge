import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import cyipopt_available
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP
def test_model1_CyIpoptNLP(self):
    model = create_model1()
    nlp = PyomoNLP(model)
    cynlp = CyIpoptNLP(nlp)
    self._check_model1(nlp, cynlp)