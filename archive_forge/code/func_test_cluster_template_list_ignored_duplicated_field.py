from unittest import mock
from magnumclient.common.apiclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.cluster_templates import ClusterTemplate
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.list')
def test_cluster_template_list_ignored_duplicated_field(self, mock_list):
    mock_list.return_value = [FakeClusterTemplate()]
    self._test_arg_success('cluster-template-list --fields coe,coe,coe,name,name', keyword='\n| uuid | name | Coe |\n')
    expected_args = self._get_expected_args_list()
    mock_list.assert_called_once_with(**expected_args)