import math
import os
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.param import _ParamData
from pyomo.core.base.set import _SetData
from pyomo.core.base.units_container import units, pint_available, UnitsError
from io import StringIO
def test_mutable_abs_expr(self):
    model = ConcreteModel()
    model.P = Param([1, 2], initialize=-1.0, mutable=True)
    model.Q = Param([1, 2], default=-1.0, mutable=True)
    model.R = Param([1, 2], mutable=True)
    model.R[1] = -1.0
    model.R[2] = -1.0
    model.x = Var()
    model.CON1 = Constraint(expr=abs(model.P[1]) <= model.x)
    model.CON2 = Constraint(expr=abs(model.Q[1]) <= model.x)
    model.CON3 = Constraint(expr=abs(model.R[1]) <= model.x)
    self.assertEqual(1.0, value(model.CON1[None].lower))
    self.assertEqual(1.0, value(model.CON2[None].lower))
    self.assertEqual(1.0, value(model.CON3[None].lower))
    model.P[1] = -3.0
    model.Q[1] = -3.0
    model.R[1] = -3.0
    self.assertEqual(3.0, value(model.CON1[None].lower))
    self.assertEqual(3.0, value(model.CON2[None].lower))
    self.assertEqual(3.0, value(model.CON3[None].lower))