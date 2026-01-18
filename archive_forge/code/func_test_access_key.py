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
def test_access_key(self):
    t = template_format.parse(user_accesskey_template)
    stack = utils.parse_stack(t)
    self.create_user(t, stack, 'CfnUser')
    rsrc = self.create_access_key(t, stack, 'HostKeys')
    self.assertEqual(self.fc.access, rsrc.resource_id)
    self.assertEqual(self.fc.secret, rsrc._secret)
    rs_data = resource_data_object.ResourceData.get_all(rsrc)
    self.assertEqual(self.fc.secret, rs_data.get('secret_key'))
    self.assertEqual(self.fc.credential_id, rs_data.get('credential_id'))
    self.assertEqual(2, len(rs_data.keys()))
    self.assertEqual(utils.PhysName(stack.name, 'CfnUser'), rsrc.FnGetAtt('UserName'))
    rsrc._secret = None
    self.assertEqual(self.fc.secret, rsrc.FnGetAtt('SecretAccessKey'))
    self.assertRaises(exception.InvalidTemplateAttribute, rsrc.FnGetAtt, 'Foo')
    scheduler.TaskRunner(rsrc.delete)()
    self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)