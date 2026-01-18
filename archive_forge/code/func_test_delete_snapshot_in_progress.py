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
def test_delete_snapshot_in_progress(self, mock_load):
    stk = self._create_stack('test_delete_snapshot_in_progress')
    mock_load.return_value = stk
    snapshot = mock.Mock()
    snapshot.id = str(uuid.uuid4())
    snapshot.status = 'IN_PROGRESS'
    self.patchobject(snapshot_objects.Snapshot, 'get_snapshot_by_stack').return_value = snapshot
    ex = self.assertRaises(dispatcher.ExpectedException, self.engine.delete_snapshot, self.ctx, stk.identifier(), snapshot.id)
    msg = 'Deleting in-progress snapshot is not supported'
    self.assertIn(msg, str(ex.exc_info[1]))
    self.assertEqual(exception.NotSupported, ex.exc_info[0])