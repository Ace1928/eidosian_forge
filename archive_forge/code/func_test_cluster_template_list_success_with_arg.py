from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.list')
def test_cluster_template_list_success_with_arg(self, mock_list):
    self._test_arg_success('cluster-template-list --limit 1 --sort-dir asc --sort-key uuid')
    expected_args = self._get_expected_args_list(1, 'asc', 'uuid')
    mock_list.assert_called_once_with(**expected_args)