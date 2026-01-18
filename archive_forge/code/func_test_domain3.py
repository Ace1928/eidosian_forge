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
@unittest.expectedFailure
def test_domain3(self):

    def x_domain(model):
        yield NonNegativeReals
        yield Reals
        yield Integers
    self.model.x = VarList(domain=x_domain)
    self.instance = self.model.create_instance()
    self.instance.x.add()
    self.instance.x.add()
    self.instance.x.add()
    self.assertEqual(self.instance.x.domain, None)
    self.assertEqual(str(self.instance.x[0].domain), str(NonNegativeReals))
    self.assertEqual(str(self.instance.x[1].domain), str(Reals))
    self.assertEqual(str(self.instance.x[2].domain), str(Integers))
    try:
        self.instance.x.domain = Reals
    except AttributeError:
        pass
    self.assertEqual(self.instance.x.domain, None)