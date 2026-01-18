from unittest import mock
from magnumclient.common import cliutils
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1 import certificates_shell
@mock.patch('magnumclient.v1.clusters.ClusterManager.get')
@mock.patch('magnumclient.v1.certificates.CertificateManager.get')
def test_cluster_ca_show_success(self, mock_cert_get, mock_cluster_get):
    mockcluster = mock.MagicMock()
    mockcluster.status = 'CREATE_COMPLETE'
    mockcluster.uuid = 'xxx'
    mock_cluster_get.return_value = mockcluster
    self._test_arg_success('ca-show xxx')
    expected_args = {}
    expected_args['cluster_uuid'] = mockcluster.uuid
    mock_cert_get.assert_called_once_with(**expected_args)