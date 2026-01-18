from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.get')
@mock.patch('magnumclient.v1.clusters.ClusterManager.get')
def test_cluster_show_clustertemplate_metadata(self, mock_cluster, mock_clustertemplate):
    mock_cluster.return_value = mock.MagicMock(cluster_template_id=0)
    mock_clustertemplate.return_value = test_clustertemplates_shell.FakeClusterTemplate(info={'links': 0, 'uuid': 0, 'id': 0, 'name': ''})
    self._test_arg_success('cluster-show --long x')
    mock_cluster.assert_called_once_with('x')
    mock_clustertemplate.assert_called_once_with(0)