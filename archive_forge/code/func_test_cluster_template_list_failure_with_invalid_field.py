from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.list')
def test_cluster_template_list_failure_with_invalid_field(self, mock_list):
    mock_list.return_value = [FakeClusterTemplate()]
    _error_msg = [".*?^Non-existent fields are specified: ['xxx','zzz']"]
    self.assertRaises(exceptions.CommandError, self._test_arg_failure, 'cluster-template-list --fields xxx,coe,zzz', _error_msg)
    expected_args = self._get_expected_args_list()
    mock_list.assert_called_once_with(**expected_args)