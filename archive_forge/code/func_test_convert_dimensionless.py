import math
import pickle
from pyomo.common.errors import PyomoException
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.util.check_units import assert_units_consistent, check_units_equivalent
from pyomo.core.expr import inequality
from pyomo.core.expr.numvalue import NumericConstant
import pyomo.core.expr as EXPR
from pyomo.core.base.units_container import (
from io import StringIO
def test_convert_dimensionless(self):
    u = units
    m = ConcreteModel()
    m.x = Var()
    foo = u.convert(m.x, to_units=u.dimensionless)
    foo = u.convert(m.x, to_units=None)
    foo = u.convert(m.x, to_units=1.0)
    with self.assertRaises(InconsistentUnitsError):
        foo = u.convert(m.x, to_units=u.kg)
    m.y = Var(units=u.kg)
    with self.assertRaises(InconsistentUnitsError):
        foo = u.convert(m.y, to_units=u.dimensionless)
    with self.assertRaises(InconsistentUnitsError):
        foo = u.convert(m.y, to_units=None)
    with self.assertRaises(InconsistentUnitsError):
        foo = u.convert(m.y, to_units=1.0)