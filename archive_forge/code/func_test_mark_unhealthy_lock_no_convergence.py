from unittest import mock
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.common import identifier
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import dependencies
from heat.engine import resource as res
from heat.engine.resources.aws.ec2 import instance as ins
from heat.engine import service
from heat.engine import stack
from heat.engine import stack_lock
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
@tools.stack_context('service_mark_unhealthy_lock_no_converge_test_stk')
def test_mark_unhealthy_lock_no_convergence(self):
    mock_acquire = self.patchobject(stack_lock.StackLock, 'acquire', return_value=None)
    mock_release = self.patchobject(stack_lock.StackLock, 'release', return_value=None)
    self.eng.resource_mark_unhealthy(self.ctx, self.stack.identifier(), 'WebServer', True, resource_status_reason='')
    mock_acquire.assert_called_once_with()
    mock_release.assert_called_once_with()