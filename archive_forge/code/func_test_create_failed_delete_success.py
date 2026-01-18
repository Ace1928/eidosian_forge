from unittest import mock
import uuid
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.aws.ec2 import subnet as sn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_create_failed_delete_success(self):
    stack_name = 'test_subnet_'
    self._mock_create_subnet_failed(stack_name)
    t = template_format.parse(self.test_template)
    tmpl = template.Template(t)
    stack = parser.Stack(utils.dummy_context(), stack_name, tmpl, stack_id=str(uuid.uuid4()))
    tmpl.t['Resources']['the_subnet']['Properties']['VpcId'] = 'aaaa'
    resource_defns = tmpl.resource_definitions(stack)
    rsrc = sn.Subnet('the_subnet', resource_defns['the_subnet'], stack)
    rsrc.validate()
    self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
    self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
    ref_id = rsrc.FnGetRefId()
    self.assertEqual(u'cccc', ref_id)
    self.mockclient.create_subnet.assert_called_once_with({'subnet': {'network_id': u'aaaa', 'cidr': u'10.0.0.0/24', 'ip_version': 4, 'name': self.subnet_name}})
    self.assertEqual(1, self.mockclient.show_network.call_count)
    self.assertIsNone(scheduler.TaskRunner(rsrc.delete)())
    self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
    self.assertEqual(2, self.mockclient.show_network.call_count)
    self.mockclient.delete_subnet.assert_called_once_with('cccc')