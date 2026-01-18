from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import node_data
from heat.engine.resources.aws.iam import user
from heat.engine.resources.openstack.heat import access_policy as ap
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.objects import resource_data as resource_data_object
from heat.tests import common
from heat.tests import utils
def test_access_key_no_user(self):
    t = template_format.parse(user_accesskey_template)
    t['Resources']['HostKeys']['Properties']['UserName'] = 'NonExistent'
    stack = utils.parse_stack(t)
    stack['CfnUser'].resource_id = self.fc.user_id
    resource_defns = stack.t.resource_definitions(stack)
    rsrc = user.AccessKey('HostKeys', resource_defns['HostKeys'], stack)
    create = scheduler.TaskRunner(rsrc.create)
    self.assertRaises(exception.ResourceFailure, create)
    self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
    scheduler.TaskRunner(rsrc.delete)()
    self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)