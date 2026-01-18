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
def test_get_valueattr(self):
    self.assertEqual(self.instance.A._value, self.sparse_data.get(None, NoValue))
    if self.data.get(None, 0) is NoValue:
        try:
            value(self.instance.A)
            self.fail('Expected value error')
        except ValueError:
            pass
    else:
        self.assertEqual(self.instance.A.value, self.data[None])