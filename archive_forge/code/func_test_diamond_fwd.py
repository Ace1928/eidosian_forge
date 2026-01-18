from heat.common import exception
from heat.engine import dependencies
from heat.tests import common
def test_diamond_fwd(self):
    self._dep_test_fwd(('last', 'mid1'), ('last', 'mid2'), ('mid1', 'first'), ('mid2', 'first'))