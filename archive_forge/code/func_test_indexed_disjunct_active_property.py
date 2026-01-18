from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core import ConcreteModel, Var, Constraint
from pyomo.gdp import Disjunction, Disjunct
from pyomo.gdp.disjunct import AutoLinkedBooleanVar, AutoLinkedBinaryVar
def test_indexed_disjunct_active_property(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 12))

    @m.Disjunct([0, 1, 2])
    def disjunct(d, i):
        m = d.model()
        if i == 0:
            d.cons = Constraint(expr=m.x >= 3)
        elif i == 1:
            d.cons = Constraint(expr=m.x >= 8)
        else:
            d.cons = Constraint(expr=m.x == 12)
    self.assertTrue(m.disjunct.active)
    m.disjunct[1].deactivate()
    self.assertTrue(m.disjunct.active)
    m.disjunct[0].deactivate()
    m.disjunct[2].deactivate()
    self.assertFalse(m.disjunct.active)
    m.disjunct.activate()
    self.assertTrue(m.disjunct.active)
    m.disjunct.deactivate()
    self.assertFalse(m.disjunct.active)
    for i in range(3):
        self.assertFalse(m.disjunct[i].active)