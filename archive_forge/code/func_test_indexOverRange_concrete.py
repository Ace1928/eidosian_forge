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
def test_indexOverRange_concrete(self):
    inst = ConcreteModel()
    inst.p = Param(range(1, 3), range(2), initialize=1.0)
    self.assertEqual(sorted(inst.p.keys()), [(1, 0), (1, 1), (2, 0), (2, 1)])
    self.assertEqual(inst.p[1, 0], 1.0)
    self.assertRaises(KeyError, inst.p.__getitem__, (0, 0))