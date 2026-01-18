from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging import conffixture
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.common import template_format
from heat.engine import service
from heat.engine import stack
from heat.objects import snapshot as snapshot_objects
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch.object(stack.Stack, 'load')
def test_delete_snapshot_not_found(self, mock_load):
    stk = self._create_stack('stack_snapshot_delete_not_found')
    mock_load.return_value = stk
    snapshot_id = str(uuid.uuid4())
    ex = self.assertRaises(dispatcher.ExpectedException, self.engine.delete_snapshot, self.ctx, stk.identifier(), snapshot_id)
    self.assertEqual(exception.NotFound, ex.exc_info[0])
    mock_load.assert_called_once_with(self.ctx, stack=mock.ANY)