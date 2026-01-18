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
def test_dimen1(self):
    model = AbstractModel()
    model.A = Set(dimen=2, initialize=[(1, 2), (3, 4)])
    model.B = Set(dimen=3, initialize=[(1, 1, 1), (2, 2, 2), (3, 3, 3)])
    model.C = Set(dimen=1, initialize=[9, 8, 7, 6, 5])
    model.x = Param(model.A, model.B, model.C, initialize=-1)
    model.y = Param(model.B, initialize=1)
    instance = model.create_instance()
    self.assertEqual(instance.x.dim(), 6)
    self.assertEqual(instance.y.dim(), 3)