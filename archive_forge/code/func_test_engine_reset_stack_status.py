from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from oslo_serialization import jsonutils as json
from heat.common import context
from heat.common import environment_util as env_util
from heat.common import exception
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.engine.cfn import template as cfntemplate
from heat.engine import environment
from heat.engine.hot import functions as hot_functions
from heat.engine.hot import template as hottemplate
from heat.engine import resource as res
from heat.engine import service
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
@mock.patch('heat.engine.service.ThreadGroupManager', return_value=mock.Mock())
@mock.patch.object(stack_object.Stack, 'get_all')
@mock.patch.object(stack_object.Stack, 'get_by_id')
@mock.patch('heat.engine.stack_lock.StackLock', return_value=mock.Mock())
@mock.patch.object(parser.Stack, 'load')
@mock.patch.object(context, 'get_admin_context')
def test_engine_reset_stack_status(self, mock_admin_context, mock_stack_load, mock_stacklock, mock_get_by_id, mock_get_all, mock_thread):
    mock_admin_context.return_value = self.ctx
    db_stack = mock.MagicMock()
    db_stack.id = 'foo'
    db_stack.status = 'IN_PROGRESS'
    db_stack.status_reason = None
    unlocked_stack = mock.MagicMock()
    unlocked_stack.id = 'bar'
    unlocked_stack.status = 'IN_PROGRESS'
    unlocked_stack.status_reason = None
    unlocked_stack_failed = mock.MagicMock()
    unlocked_stack_failed.id = 'bar'
    unlocked_stack_failed.status = 'FAILED'
    unlocked_stack_failed.status_reason = 'because'
    mock_get_all.return_value = [db_stack, unlocked_stack]
    mock_get_by_id.side_effect = [db_stack, unlocked_stack_failed]
    fake_stack = mock.MagicMock()
    fake_stack.action = 'CREATE'
    fake_stack.id = 'foo'
    fake_stack.status = 'IN_PROGRESS'
    mock_stack_load.return_value = fake_stack
    lock1 = mock.MagicMock()
    lock1.get_engine_id.return_value = 'old-engine'
    lock1.acquire.return_value = None
    lock2 = mock.MagicMock()
    lock2.acquire.return_value = None
    mock_stacklock.side_effect = [lock1, lock2]
    self.eng.thread_group_mgr = mock_thread
    self.eng.reset_stack_status()
    mock_admin_context.assert_called()
    filters = {'status': parser.Stack.IN_PROGRESS, 'convergence': False}
    mock_get_all.assert_called_once_with(self.ctx, filters=filters, show_nested=True)
    mock_get_by_id.assert_has_calls([mock.call(self.ctx, 'foo'), mock.call(self.ctx, 'bar')])
    mock_stack_load.assert_called_once_with(self.ctx, stack=db_stack)
    self.assertTrue(lock2.release.called)
    reason = 'Engine went down during stack %s' % fake_stack.action
    mock_thread.start_with_acquired_lock.assert_called_once_with(fake_stack, lock1, fake_stack.reset_stack_and_resources_in_progress, reason)