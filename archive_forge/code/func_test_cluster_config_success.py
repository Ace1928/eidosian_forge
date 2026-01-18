from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.get')
@mock.patch('magnumclient.v1.clusters.ClusterManager.get')
def test_cluster_config_success(self, mock_cluster, mock_clustertemplate):
    mock_cluster.return_value = FakeCluster(status='UPDATE_COMPLETE')
    self._test_arg_success('cluster-config xxx')
    mock_cluster.assert_called_with('xxx')
    mock_cluster.return_value = FakeCluster(status='CREATE_COMPLETE')
    self._test_arg_success('cluster-config xxx')
    mock_cluster.assert_called_with('xxx')
    self._test_arg_success('cluster-config --dir /tmp xxx')
    mock_cluster.assert_called_with('xxx')
    self._test_arg_success('cluster-config --force xxx')
    mock_cluster.assert_called_with('xxx')
    self._test_arg_success('cluster-config --dir /tmp --force xxx')
    mock_cluster.assert_called_with('xxx')