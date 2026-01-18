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
def test_as_quantity_expression(self):
    _pint = units._pint_registry
    Quantity = _pint.Quantity
    m = ConcreteModel()
    m.x = Var(initialize=1)
    m.y = Var(initialize=2, units=units.g)
    m.p = Param(initialize=3)
    m.q = Param(initialize=4, units=1 / units.s)
    q = as_quantity(m.x * m.p)
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q, 3 * _pint.dimensionless)
    q = as_quantity(m.x * m.q)
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q, 4 / _pint.s)
    q = as_quantity(m.y * m.p)
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q, 6 * _pint.g)
    q = as_quantity(m.y * m.q)
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q, 8 * _pint.g / _pint.s)
    q = as_quantity(m.y <= 2 * m.y)
    self.assertIs(q.__class__, bool)
    self.assertEqual(q, True)
    q = as_quantity(m.y >= 2 * m.y)
    self.assertIs(q.__class__, bool)
    self.assertEqual(q, False)
    q = as_quantity(EXPR.Expr_if(IF=m.y <= 2 * m.y, THEN=m.x, ELSE=m.p))
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q, 1 * _pint.dimensionless)
    q = as_quantity(EXPR.Expr_if(IF=m.y >= 2 * m.y, THEN=m.x, ELSE=m.p))
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q, 3 * _pint.dimensionless)
    q = as_quantity(EXPR.Expr_if(IF=m.x <= 2 * m.x, THEN=m.y, ELSE=m.q))
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q, 2 * _pint.g)
    q = as_quantity(EXPR.Expr_if(IF=m.x >= 2 * m.x, THEN=m.y, ELSE=m.q))
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q, 4 / _pint.s)
    q = as_quantity(acos(m.x))
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q.units, _pint.radian)
    self.assertEqual(q, 0 * _pint.radian)
    q = as_quantity(cos(m.x * math.pi))
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q.units, _pint.dimensionless)
    self.assertAlmostEqual(q, -1 * _pint.dimensionless)

    def MyAdder(x, y):
        return x + y
    m.EF = ExternalFunction(MyAdder, units=units.kg)
    ef = m.EF(m.x, m.y)
    q = as_quantity(ef)
    self.assertIs(q.__class__, Quantity)
    self.assertAlmostEqual(q, 3 * _pint.kg)