from unittest import mock
from heat.common import exception
from heat.common import service_utils
from heat.engine import stack_lock
from heat.objects import stack as stack_object
from heat.objects import stack_lock as stack_lock_object
from heat.tests import common
from heat.tests import utils
def test_failed_acquire_existing_lock_engine_alive(self):
    mock_create = self.patchobject(stack_lock_object.StackLock, 'create', return_value='fake-engine-id')
    slock = stack_lock.StackLock(self.context, self.stack_id, self.engine_id)
    self.patchobject(service_utils, 'engine_alive', return_value=True)
    self.assertRaises(exception.ActionInProgress, slock.acquire)
    self.mock_get_by_id.assert_called_once_with(self.context, self.stack_id, show_deleted=True, eager_load=False)
    mock_create.assert_called_once_with(self.context, self.stack_id, self.engine_id)