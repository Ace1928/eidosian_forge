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
def test_inf_bounds_on_expr(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-1, 1))
    m.y = pyo.Var()
    m.z = pyo.Var()
    m.c = pyo.Constraint(expr=m.x + m.y == m.z)
    self.tightener(m)
    self.assertEqual(m.z.lb, None)
    self.assertEqual(m.z.ub, None)