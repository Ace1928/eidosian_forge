from unittest import mock
import mistralclient.tests.unit.base_shell_test as base
@mock.patch('mistralclient.api.client.determine_client_version')
def test_no_mistral_version(self, client_mock):
    self.shell('workbook-list')
    self.assertTrue(client_mock.called)
    mistral_version = client_mock.call_args
    self.assertEqual('v2', mistral_version[0][0])