from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
@mock.patch('magnumclient.v1.clusters.ClusterManager.update')
def test_cluster_update_failure_few_args(self, mock_update):
    _error_msg = ['.*?^usage: magnum cluster-update ', '.*?^error: (the following arguments|too few arguments)', ".*?^Try 'magnum help cluster-update' for more information."]
    self._test_arg_failure('cluster-update', _error_msg)
    mock_update.assert_not_called()
    self._test_arg_failure('cluster-update test', _error_msg)
    mock_update.assert_not_called()
    self._test_arg_failure('cluster-update test add', _error_msg)
    mock_update.assert_not_called()