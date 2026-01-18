import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import cyipopt_available
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP
def test_model1_CyIpoptNLP_scaling(self):
    m = create_model1()
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.o] = 1e-06
    m.scaling_factor[m.c] = 2.0
    m.scaling_factor[m.d] = 3.0
    m.scaling_factor[m.x[1]] = 4.0
    cynlp = CyIpoptNLP(PyomoNLP(m))
    obj_scaling, x_scaling, g_scaling = cynlp.scaling_factors()
    self.assertTrue(obj_scaling == 1e-06)
    self.assertTrue(len(x_scaling) == 3)
    self.assertTrue(x_scaling[0] == 1.0)
    self.assertTrue(x_scaling[1] == 1.0)
    self.assertTrue(x_scaling[2] == 4.0)
    self.assertTrue(len(g_scaling) == 2)
    self.assertTrue(g_scaling[0] == 3.0)
    self.assertTrue(g_scaling[1] == 2.0)
    m = create_model1()
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.c] = 2.0
    m.scaling_factor[m.d] = 3.0
    m.scaling_factor[m.x[1]] = 4.0
    cynlp = CyIpoptNLP(PyomoNLP(m))
    obj_scaling, x_scaling, g_scaling = cynlp.scaling_factors()
    self.assertTrue(obj_scaling == 1.0)
    self.assertTrue(len(x_scaling) == 3)
    self.assertTrue(x_scaling[0] == 1.0)
    self.assertTrue(x_scaling[1] == 1.0)
    self.assertTrue(x_scaling[2] == 4.0)
    self.assertTrue(len(g_scaling) == 2)
    self.assertTrue(g_scaling[0] == 3.0)
    self.assertTrue(g_scaling[1] == 2.0)
    m = create_model1()
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.o] = 1e-06
    m.scaling_factor[m.c] = 2.0
    m.scaling_factor[m.d] = 3.0
    cynlp = CyIpoptNLP(PyomoNLP(m))
    obj_scaling, x_scaling, g_scaling = cynlp.scaling_factors()
    self.assertTrue(obj_scaling == 1e-06)
    self.assertTrue(len(x_scaling) == 3)
    self.assertTrue(x_scaling[0] == 1.0)
    self.assertTrue(x_scaling[1] == 1.0)
    self.assertTrue(x_scaling[2] == 1.0)
    self.assertTrue(len(g_scaling) == 2)
    self.assertTrue(g_scaling[0] == 3.0)
    self.assertTrue(g_scaling[1] == 2.0)
    m = create_model1()
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.o] = 1e-06
    m.scaling_factor[m.d] = 3.0
    m.scaling_factor[m.x[1]] = 4.0
    cynlp = CyIpoptNLP(PyomoNLP(m))
    obj_scaling, x_scaling, g_scaling = cynlp.scaling_factors()
    self.assertTrue(obj_scaling == 1e-06)
    self.assertTrue(len(x_scaling) == 3)
    self.assertTrue(x_scaling[0] == 1.0)
    self.assertTrue(x_scaling[1] == 1.0)
    self.assertTrue(x_scaling[2] == 4.0)
    self.assertTrue(len(g_scaling) == 2)
    self.assertTrue(g_scaling[0] == 3.0)
    self.assertTrue(g_scaling[1] == 1.0)
    m = create_model1()
    cynlp = CyIpoptNLP(PyomoNLP(m))
    obj_scaling, x_scaling, g_scaling = cynlp.scaling_factors()
    self.assertTrue(obj_scaling is None)
    self.assertTrue(x_scaling is None)
    self.assertTrue(g_scaling is None)