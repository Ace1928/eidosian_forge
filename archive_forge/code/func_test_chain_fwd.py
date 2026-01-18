from heat.common import exception
from heat.engine import dependencies
from heat.tests import common
def test_chain_fwd(self):
    self._dep_test_fwd(('third', 'second'), ('second', 'first'))