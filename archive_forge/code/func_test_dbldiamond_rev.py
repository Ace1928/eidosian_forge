from heat.common import exception
from heat.engine import dependencies
from heat.tests import common
def test_dbldiamond_rev(self):
    self._dep_test_rev(('last', 'a1'), ('last', 'a2'), ('a1', 'b1'), ('a2', 'b1'), ('a2', 'b2'), ('b1', 'first'), ('b2', 'first'))