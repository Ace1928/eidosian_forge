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
def test_index4(self):
    self.model.A = Set(initialize=range(0, 4))

    @set_options(within=Integers)
    def B_index(model):
        return [i / 2.0 for i in model.A]

    def B_init(model, i, j):
        if j:
            return 2 + i
        return -(2 + i)
    self.model.B = Param(B_index, [True, False], initialize=B_init)
    try:
        self.instance = self.model.create_instance()
        self.fail('Expected ValueError because B_index returns invalid index values')
    except ValueError:
        pass