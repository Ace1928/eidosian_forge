import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
def create_pyomo_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(3), bounds=(-10, 10), initialize={0: 1.0, 1: 2.0, 2: 4.0})
    m.obj = pyo.Objective(expr=m.x[0] ** 2 + m.x[0] * m.x[1] + m.x[0] * m.x[2] + m.x[2] ** 2)
    m.con1 = pyo.Constraint(expr=m.x[0] * m.x[1] + m.x[0] * m.x[2] == 4)
    m.con2 = pyo.Constraint(expr=m.x[0] + m.x[2] == 4)
    return m