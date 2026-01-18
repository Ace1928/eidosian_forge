from unittest import mock
from magnumclient.common import cliutils
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1 import certificates_shell
@mock.patch('magnumclient.v1.clusters.ClusterManager.get')
@mock.patch('magnumclient.v1.certificates.CertificateManager.rotate_ca')
def test_ca_rotate_no_cluster_arg(self, mock_rotate_ca, mock_cluster_get):
    _error_msg = ['.*(error: argument --cluster is required|error: the following arguments are required: --cluster).*', ".*Try 'magnum help ca-rotate' for more information.*"]
    self._test_arg_failure('ca-rotate', _error_msg)
    mock_rotate_ca.assert_not_called()
    mock_cluster_get.assert_not_called()