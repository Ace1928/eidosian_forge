from unittest import mock
import mistralclient.tests.unit.base_shell_test as base
@mock.patch('mistralclient.api.client.client')
def test_auth_url(self, client_mock):
    self.shell('--os-auth-url=https://127.0.0.1:35357/v3 --os-username=admin --os-password=1234 workbook-list')
    self.assertTrue(client_mock.called)
    params = client_mock.call_args
    self.assertEqual('https://127.0.0.1:35357/v3', params[1]['auth_url'])