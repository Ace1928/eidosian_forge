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
def test_module_example(self):
    from pyomo.environ import ConcreteModel, Var, Objective, units
    model = ConcreteModel()
    model.acc = Var()
    model.obj = Objective(expr=(model.acc * units.m / units.s ** 2 - 9.81 * units.m / units.s ** 2) ** 2)
    self.assertEqual('m**2/s**4', str(units.get_units(model.obj.expr)))