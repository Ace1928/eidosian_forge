from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
@mock.patch('magnumclient.v1.clusters.ClusterManager.get')
def test_cluster_config_failure_wrong_arg(self, mock_cluster):
    self._test_arg_failure('cluster-config xxx yyy', self._unrecognized_arg_error)
    mock_cluster.assert_not_called()