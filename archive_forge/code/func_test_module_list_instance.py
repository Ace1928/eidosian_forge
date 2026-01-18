from unittest import mock
import testtools
from troveclient import base
from troveclient.v1 import instances
def test_module_list_instance(self):
    resp = mock.Mock()
    resp.status_code = 200
    body = {'modules': []}
    self.instances.api.client.get = mock.Mock(return_value=(resp, body))
    self.instances.modules(self.instance_with_id)
    resp.status_code = 500
    self.assertRaises(Exception, self.instances.modules, 'instance1')