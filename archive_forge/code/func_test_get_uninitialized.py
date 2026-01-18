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
def test_get_uninitialized(self):
    model = AbstractModel()
    model.a = Param()
    model.b = Set(initialize=[1, 2, 3])
    model.c = Param(model.b, initialize=2, within=Reals)
    instance = model.create_instance()
    self.assertRaises(ValueError, value, instance.a)