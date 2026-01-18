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
def createNonIndexedParamMethod(func, init_xy, new_xy, tol=1e-10):

    def testMethod(self):
        model = ConcreteModel()
        model.Q1 = Param(initialize=init_xy[0], mutable=True)
        model.x = Var()
        model.CON = Constraint(expr=func(model.Q1) <= model.x)
        self.assertAlmostEqual(init_xy[1], value(model.CON[None].lower), delta=1e-10)
        model.Q1 = new_xy[0]
        self.assertAlmostEqual(new_xy[1], value(model.CON[None].lower), delta=tol)
    return testMethod