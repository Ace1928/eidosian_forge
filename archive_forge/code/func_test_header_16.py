import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.environ as pe
from pyomo.contrib import appsi
from pyomo.contrib.appsi.cmodel import cmodel_available
import os
def test_header_16(self):
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.y = pe.Var()
    m.obj = pe.Objective(expr=m.x ** 2 + m.y ** 2)
    m.c = pe.Constraint(expr=m.x ** 2 + m.y ** 2 == 1)
    correct_lines = ['g3 1 1 0', '2 1 1 0 1', '1 1', '0 0', '2 2 2', '0 0 0 1', '0 0 0 0 0', '2 2', '0 0', '0 0 0 0 0']
    self._write_and_check_header(m, correct_lines)