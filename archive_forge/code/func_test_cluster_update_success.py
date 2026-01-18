from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
@mock.patch('magnumclient.v1.clusters.ClusterManager.update')
def test_cluster_update_success(self, mock_update):
    self._test_arg_success('cluster-update test add test=test')
    patch = [{'op': 'add', 'path': '/test', 'value': 'test'}]
    mock_update.assert_called_once_with('test', patch, False)