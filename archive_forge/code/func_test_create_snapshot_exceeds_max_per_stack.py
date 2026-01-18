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
def test_create_snapshot_exceeds_max_per_stack(self, mock_load):
    stk = self._create_stack('stack_snapshot_exceeds_max')
    mock_load.return_value = stk
    cfg.CONF.set_override('max_snapshots_per_stack', 0)
    ex = self.assertRaises(dispatcher.ExpectedException, self.engine.stack_snapshot, self.ctx, stk.identifier(), 'snap_none')
    self.assertEqual(exception.RequestLimitExceeded, ex.exc_info[0])
    self.assertIn('You have reached the maximum snapshots per stack', str(ex.exc_info[1]))