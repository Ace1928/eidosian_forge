from os.path import abspath, dirname, join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import import_file
from pyomo.contrib.satsolver.satsolver import satisfiable, z3_available
from pyomo.core.base.set_types import PositiveIntegers, NonNegativeReals, Binary
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
def test_bounds_sat(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 5))
    m.c1 = Constraint(expr=4.99 == m.x)
    m.o = Objective(expr=m.x)
    self.assertTrue(satisfiable(m))