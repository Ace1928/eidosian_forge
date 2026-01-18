import sys
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var
from pyomo.common.modeling import unique_component_name, NOTSET
def test_NOTSET(self):
    self.assertEqual(str(NOTSET), 'NOTSET')
    assert 'sphinx' not in sys.modules
    self.assertEqual(repr(NOTSET), 'pyomo.common.modeling.NOTSET')