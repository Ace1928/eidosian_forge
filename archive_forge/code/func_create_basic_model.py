from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.utils import (
import pyomo.environ as pyo
import numpy as np
def create_basic_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3], domain=pyo.Reals)
    for i in range(1, 4):
        m.x[i].value = i
    m.c1 = pyo.Constraint(expr=m.x[1] ** 2 - m.x[2] - 1 == 0)
    m.c2 = pyo.Constraint(expr=m.x[1] - m.x[3] - 0.5 == 0)
    m.d1 = pyo.Constraint(expr=m.x[1] + m.x[2] <= 100.0)
    m.d2 = pyo.Constraint(expr=m.x[2] + m.x[3] >= -100.0)
    m.d3 = pyo.Constraint(expr=m.x[2] + m.x[3] + m.x[1] >= -500.0)
    m.x[2].setlb(0.0)
    m.x[3].setlb(0.0)
    m.x[2].setub(100.0)
    m.obj = pyo.Objective(expr=m.x[2] ** 2)
    return m