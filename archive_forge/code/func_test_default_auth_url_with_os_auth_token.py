from unittest import mock
import mistralclient.tests.unit.base_shell_test as base
@mock.patch('mistralclient.api.client.client')
def test_default_auth_url_with_os_auth_token(self, client_mock):
    self.shell('--os-auth-token=abcd1234 workbook-list')
    self.assertTrue(client_mock.called)
    params = client_mock.call_args
    self.assertEqual('http://localhost:35357/v3', params[1]['auth_url'])