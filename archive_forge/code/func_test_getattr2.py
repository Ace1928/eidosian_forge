import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.base import IntegerSet
from pyomo.core.expr.numeric_expr import (
from pyomo.core.staleflag import StaleFlagManager
from pyomo.environ import (
from pyomo.core.base.units_container import units, pint_available, UnitsError
def test_getattr2(self):
    """
        Verify the behavior of non-standard suffixes with an array of variables
        """
    model = AbstractModel()
    model.X = Set(initialize=[1, 3, 5])
    model.a = Var(model.X)
    model.suffix = Suffix(datatype=Suffix.INT)
    try:
        self.assertEqual(model.a.suffix, None)
        self.fail('Expected AttributeError')
    except AttributeError:
        pass
    instance = model.create_instance()
    self.assertEqual(instance.suffix.get(instance.a[1]), None)
    instance.suffix.set_value(instance.a[1], True)
    self.assertEqual(instance.suffix.get(instance.a[1]), True)