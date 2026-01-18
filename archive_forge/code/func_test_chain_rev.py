from heat.common import exception
from heat.engine import dependencies
from heat.tests import common
def test_chain_rev(self):
    self._dep_test_rev(('third', 'second'), ('second', 'first'))