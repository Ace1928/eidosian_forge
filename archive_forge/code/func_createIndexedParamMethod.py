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
def createIndexedParamMethod(func, init_xy, new_xy, tol=1e-10):

    def testMethod(self):
        model = ConcreteModel()
        model.P = Param([1, 2], initialize=init_xy[0], mutable=True)
        model.Q = Param([1, 2], default=init_xy[0], mutable=True)
        model.R = Param([1, 2], mutable=True)
        model.R[1] = init_xy[0]
        model.R[2] = init_xy[0]
        model.x = Var()
        model.CON1 = Constraint(expr=func(model.P[1]) <= model.x)
        model.CON2 = Constraint(expr=func(model.Q[1]) <= model.x)
        model.CON3 = Constraint(expr=func(model.R[1]) <= model.x)
        self.assertAlmostEqual(init_xy[1], value(model.CON1[None].lower), delta=tol)
        self.assertAlmostEqual(init_xy[1], value(model.CON2[None].lower), delta=tol)
        self.assertAlmostEqual(init_xy[1], value(model.CON3[None].lower), delta=tol)
        model.P[1] = new_xy[0]
        model.Q[1] = new_xy[0]
        model.R[1] = new_xy[0]
        self.assertAlmostEqual(new_xy[1], value(model.CON1[None].lower), delta=tol)
        self.assertAlmostEqual(new_xy[1], value(model.CON2[None].lower), delta=tol)
        self.assertAlmostEqual(new_xy[1], value(model.CON3[None].lower), delta=tol)
    return testMethod