import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.fbbt.fbbt import fbbt, compute_bounds_on_expr
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.fileutils import find_library
from pyomo.common.log import LoggingIntercept
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.core.expr.numeric_expr import (
import math
import platform
from io import StringIO
def test_multiple_constraints2(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-3, 3))
    m.y = pyo.Var(bounds=(None, 0))
    m.z = pyo.Var()
    m.c = pyo.ConstraintList()
    m.c.add(-m.x - m.y >= -1)
    m.c.add(-m.x - m.y <= -1)
    m.c.add(-m.y - m.x * m.z >= -2)
    m.c.add(-m.y - m.x * m.z <= 2)
    m.c.add(-m.x - m.z == 1)
    self.tightener(m)
    self.assertAlmostEqual(pyo.value(m.x.lb), 1, 8)
    self.assertAlmostEqual(pyo.value(m.x.ub), 1, 8)
    self.assertAlmostEqual(pyo.value(m.y.lb), 0, 8)
    self.assertAlmostEqual(pyo.value(m.y.ub), 0, 8)
    self.assertAlmostEqual(pyo.value(m.z.lb), -2, 8)
    self.assertAlmostEqual(pyo.value(m.z.ub), -2, 8)