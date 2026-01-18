import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.tests.compare_utils import (
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models
def test_error_fixed_inputs_outputs(self):
    m = pyo.ConcreteModel()
    m.egb = ExternalGreyBoxBlock()
    m.egb.set_external_model(ex_models.PressureDropSingleOutput())
    m.egb.inputs['Pin'].fix(100)
    m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
    with self.assertRaises(NotImplementedError):
        pyomo_nlp = PyomoNLPWithGreyBoxBlocks(m)
    m = pyo.ConcreteModel()
    m.egb = ExternalGreyBoxBlock()
    m.egb.set_external_model(ex_models.PressureDropTwoOutputs())
    m.egb.outputs['P2'].fix(50)
    m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
    with self.assertRaises(NotImplementedError):
        pyomo_nlp = PyomoNLPWithGreyBoxBlocks(m)