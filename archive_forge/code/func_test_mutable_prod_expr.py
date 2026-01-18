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
def test_mutable_prod_expr(self):
    model = ConcreteModel()
    model.P = Param([1, 2], initialize=0.0, mutable=True)
    model.Q = Param([1, 2], default=0.0, mutable=True)
    model.R = Param([1, 2], mutable=True)
    model.R[1] = 0.0
    model.R[2] = 0.0
    model.x = Var()
    model.CON1 = Constraint(expr=model.P[1] * model.P[2] <= model.x)
    model.CON2 = Constraint(expr=model.Q[1] * model.Q[2] <= model.x)
    model.CON3 = Constraint(expr=model.R[1] * model.R[2] <= model.x)
    self.assertEqual(0.0, value(model.CON1[None].lower))
    self.assertEqual(0.0, value(model.CON2[None].lower))
    self.assertEqual(0.0, value(model.CON3[None].lower))
    model.P[1] = 3.0
    model.P[2] = 2.0
    model.Q[1] = 3.0
    model.Q[2] = 2.0
    model.R[1] = 3.0
    model.R[2] = 2.0
    self.assertEqual(6.0, value(model.CON1[None].lower))
    self.assertEqual(6.0, value(model.CON2[None].lower))
    self.assertEqual(6.0, value(model.CON3[None].lower))