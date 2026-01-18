from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_alarm_attribute(self):
    heat_plugin = self.stack.clients.client_plugin('heat')
    heat_plugin.get_heat_cfn_url = mock.Mock(return_value='http://server.test:8000/v1')
    alarm_url = self.policy.FnGetAtt('alarm_url')
    base = alarm_url.split('?')[0].split('%3A')
    self.assertEqual('http://server.test:8000/v1/signal/arn', base[0])
    self.assertEqual('openstack', base[1])
    self.assertEqual('heat', base[2])
    self.assertEqual('test_tenant_id', base[4])
    res = base[5].split('/')
    self.assertEqual('stacks', res[0])
    self.assertEqual(self.stack_name, res[1])
    self.assertEqual('resources', res[3])
    self.assertEqual('my-policy', res[4])
    args = sorted(alarm_url.split('?')[1].split('&'))
    self.assertEqual('AWSAccessKeyId', args[0].split('=')[0])
    self.assertEqual('Signature', args[1].split('=')[0])
    self.assertEqual('SignatureMethod', args[2].split('=')[0])
    self.assertEqual('SignatureVersion', args[3].split('=')[0])