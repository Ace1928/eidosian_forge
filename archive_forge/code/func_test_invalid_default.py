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
def test_invalid_default(self):
    model = ConcreteModel()
    with self.assertRaisesRegex(ValueError, 'Default value \\(-1\\) is not valid for Param p domain NonNegativeIntegers'):
        model.p = Param(default=-1, within=NonNegativeIntegers)