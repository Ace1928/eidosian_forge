from os.path import abspath, dirname, join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import import_file
from pyomo.contrib.satsolver.satsolver import satisfiable, z3_available
from pyomo.core.base.set_types import PositiveIntegers, NonNegativeReals, Binary
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
def test_multiple_disjunctions_unsat(self):
    m = ConcreteModel()
    m.x1 = Var(bounds=(0, 8))
    m.x2 = Var(bounds=(0, 8))
    m.obj = Objective(expr=m.x1 + m.x2, sense=minimize)
    m.y1 = Disjunct()
    m.y2 = Disjunct()
    m.y1.c1 = Constraint(expr=m.x1 >= 2)
    m.y1.c2 = Constraint(expr=m.x2 >= 2)
    m.y2.c1 = Constraint(expr=m.x1 >= 2)
    m.y2.c2 = Constraint(expr=m.x2 >= 2)
    m.djn1 = Disjunction(expr=[m.y1, m.y2])
    m.z1 = Disjunct()
    m.z2 = Disjunct()
    m.z1.c1 = Constraint(expr=m.x1 <= 1)
    m.z1.c2 = Constraint(expr=m.x2 <= 1)
    m.z2.c1 = Constraint(expr=m.x1 <= 1)
    m.z2.c2 = Constraint(expr=m.x2 <= 1)
    m.djn2 = Disjunction(expr=[m.z1, m.z2])
    self.assertFalse(satisfiable(m))