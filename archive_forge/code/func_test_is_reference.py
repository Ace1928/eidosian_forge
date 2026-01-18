from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.environ import (
from pyomo.core.base.set import GlobalSets
def test_is_reference(self):
    m = ConcreteModel()

    class _NotSpecified(object):
        pass
    m.comp = Component(ctype=_NotSpecified)
    self.assertFalse(m.comp.is_reference())