import itertools
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.expr.visitor import identify_variables
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sps
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models import (
def test_construct_scalar(self):
    m = pyo.ConcreteModel()
    m.ex_block = ExternalGreyBoxBlock(concrete=True)
    block = m.ex_block
    self.assertIs(type(block), ScalarExternalGreyBoxBlock)
    m_ex = _make_external_model()
    input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
    external_vars = [m_ex.x, m_ex.y]
    residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
    external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
    ex_model = ExternalPyomoModel(input_vars, external_vars, residual_cons, external_cons)
    block.set_external_model(ex_model)
    self.assertEqual(len(block.inputs), len(input_vars))
    self.assertEqual(len(block.outputs), 0)
    self.assertEqual(len(block._equality_constraint_names), 2)