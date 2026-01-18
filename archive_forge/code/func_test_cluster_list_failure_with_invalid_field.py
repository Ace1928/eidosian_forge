from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
@mock.patch('magnumclient.v1.clusters.ClusterManager.list')
def test_cluster_list_failure_with_invalid_field(self, mock_list):
    mock_list.return_value = [FakeCluster()]
    _error_msg = [".*?^Non-existent fields are specified: ['xxx','zzz']"]
    self.assertRaises(exceptions.CommandError, self._test_arg_failure, 'cluster-list --fields xxx,stack_id,zzz,status', _error_msg)
    expected_args = self._get_expected_args_list()
    mock_list.assert_called_once_with(**expected_args)