from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
@mock.patch('magnumclient.v1.clusters.ClusterManager.create')
def test_cluster_create_failure_only_cluster_create_timeout(self, mock_create):
    self._test_arg_failure('cluster-create --timeout 15', self._mandatory_arg_error)
    mock_create.assert_not_called()