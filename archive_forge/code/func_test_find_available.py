from unittest import mock
from openstack.network.v2 import floating_ip
from openstack import proxy
from openstack.tests.unit import base
def test_find_available(self):
    mock_session = mock.Mock(spec=proxy.Proxy)
    mock_session.get_filter = mock.Mock(return_value={})
    mock_session.default_microversion = None
    mock_session.session = self.cloud.session
    data = {'id': 'one', 'floating_ip_address': '10.0.0.1'}
    fake_response = mock.Mock()
    body = {floating_ip.FloatingIP.resources_key: [data]}
    fake_response.json = mock.Mock(return_value=body)
    fake_response.status_code = 200
    mock_session.get = mock.Mock(return_value=fake_response)
    result = floating_ip.FloatingIP.find_available(mock_session)
    self.assertEqual('one', result.id)
    mock_session.get.assert_called_with(floating_ip.FloatingIP.base_path, headers={'Accept': 'application/json'}, params={}, microversion=None)