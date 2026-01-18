import copy
import eventlet
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_cancel_update_with_rollback(self):
    self._test_cancel_update()