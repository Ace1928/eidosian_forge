from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
@mock.patch('magnumclient.v1.clusters.ClusterManager.delete')
def test_cluster_delete_failure_no_arg(self, mock_delete):
    self._test_arg_failure('cluster-delete', self._few_argument_error)
    mock_delete.assert_not_called()