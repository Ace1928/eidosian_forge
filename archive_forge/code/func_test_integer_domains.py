from os.path import abspath, dirname, join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import import_file
from pyomo.contrib.satsolver.satsolver import satisfiable, z3_available
from pyomo.core.base.set_types import PositiveIntegers, NonNegativeReals, Binary
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
def test_integer_domains(self):
    m = ConcreteModel()
    m.x1 = Var(domain=PositiveIntegers)
    m.c1 = Constraint(expr=m.x1 == 0.5)
    self.assertFalse(satisfiable(m))