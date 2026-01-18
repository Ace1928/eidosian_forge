from os.path import abspath, dirname, join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import import_file
from pyomo.contrib.satsolver.satsolver import satisfiable, z3_available
from pyomo.core.base.set_types import PositiveIntegers, NonNegativeReals, Binary
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
def test_binary_expressions(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.z = Var()
    m.c = Constraint(expr=0 <= m.x + m.y - m.z * m.y / m.x + 7)
    m.o = Objective(expr=m.x)
    self.assertTrue(satisfiable(m))