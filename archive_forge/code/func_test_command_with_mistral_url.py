from unittest import mock
import mistralclient.tests.unit.base_shell_test as base
@mock.patch('mistralclient.api.client.client')
def test_command_with_mistral_url(self, client_mock):
    self.shell('--os-mistral-url=http://localhost:8989/v2 workbook-list')
    self.assertTrue(client_mock.called)
    params = client_mock.call_args
    self.assertEqual('http://localhost:8989/v2', params[1]['mistral_url'])