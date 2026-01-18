from unittest import mock
import testtools
from troveclient import base
from troveclient.v1 import instances
def test_module_remove(self):
    resp = mock.Mock()
    resp.status_code = 200
    body = {'modules': []}
    self.instances.api.client.delete = mock.Mock(return_value=(resp, body))
    self.instances.module_remove(self.instance_with_id, 'mod_id')
    resp.status_code = 500
    self.assertRaises(Exception, self.instances.module_remove, 'instance1', 'mod1')