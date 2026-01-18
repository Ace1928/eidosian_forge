import copy
import eventlet
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_cancel_update_without_rollback(self):
    stack_id = self._test_cancel_update(rollback=False, expected_status='UPDATE_FAILED')
    self.assertTrue(test.call_until_true(60, 2, self.verify_resource_status, stack_id, 'test1', 'UPDATE_COMPLETE'))
    eventlet.sleep(2)
    self.assertTrue(self.verify_resource_status(stack_id, 'test2', 'CREATE_COMPLETE'))