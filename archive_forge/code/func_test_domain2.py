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
def test_domain2(self):

    def x_domain(model, i):
        if i == 1:
            return NonNegativeReals
        elif i == 2:
            return Reals
        elif i == 3:
            return Integers
    self.model.x = VarList(domain=x_domain)
    self.instance = self.model.create_instance()
    self.instance.x.add()
    self.instance.x.add()
    self.instance.x.add()
    self.assertEqual(str(self.instance.x[1].domain), str(NonNegativeReals))
    self.assertEqual(str(self.instance.x[2].domain), str(Reals))
    self.assertEqual(str(self.instance.x[3].domain), str(Integers))
    try:
        self.instance.x.domain
    except AttributeError:
        pass
    self.instance.x.domain = Binary
    self.assertEqual(str(self.instance.x[1].domain), str(Binary))
    self.assertEqual(str(self.instance.x[2].domain), str(Binary))
    self.assertEqual(str(self.instance.x[3].domain), str(Binary))