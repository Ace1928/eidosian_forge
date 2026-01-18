from unittest import mock
import mistralclient.tests.unit.base_shell_test as base
@mock.patch('mistralclient.api.client.client')
def test_no_service_type(self, client_mock):
    self.shell('workbook-list')
    self.assertTrue(client_mock.called)
    params = client_mock.call_args
    self.assertEqual('workflowv2', params[1]['service_type'])