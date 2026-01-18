from unittest import mock
from openstack.message.v2 import _proxy
from openstack.message.v2 import claim
from openstack.message.v2 import message
from openstack.message.v2 import queue
from openstack.message.v2 import subscription
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
@mock.patch.object(proxy_base.Proxy, '_get_resource')
def test_message_delete_ignore(self, mock_get_resource):
    fake_message = mock.Mock()
    fake_message.id = 'message_id'
    mock_get_resource.return_value = fake_message
    self._verify('openstack.proxy.Proxy._delete', self.proxy.delete_message, method_args=['test_queue', 'resource_or_id', None, True], expected_args=[message.Message, fake_message], expected_kwargs={'ignore_missing': True})
    self.assertIsNone(fake_message.claim_id)
    mock_get_resource.assert_called_once_with(message.Message, 'resource_or_id', queue_name='test_queue')