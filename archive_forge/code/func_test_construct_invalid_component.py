from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core import ConcreteModel, Var, Constraint
from pyomo.gdp import Disjunction, Disjunct
from pyomo.gdp.disjunct import AutoLinkedBooleanVar, AutoLinkedBinaryVar
def test_construct_invalid_component(self):
    m = ConcreteModel()
    m.d = Disjunct([1, 2])
    with self.assertRaisesRegex(ValueError, "Unexpected term for Disjunction 'dd'.\n    Expected a Disjunct object, relational expression, or iterable of\n    relational expressions but got 'IndexedDisjunct'"):
        m.dd = Disjunction(expr=[m.d])
    with self.assertRaisesRegex(ValueError, "Unexpected term for Disjunction 'ee'.\n    Expected a Disjunct object, relational expression, or iterable of\n    relational expressions but got 'str' in 'list'"):
        m.ee = Disjunction(expr=[['a']])
    with self.assertRaisesRegex(ValueError, "Unexpected term for Disjunction 'ff'.\n    Expected a Disjunct object, relational expression, or iterable of\n    relational expressions but got 'str'"):
        m.ff = Disjunction(expr=['a'])