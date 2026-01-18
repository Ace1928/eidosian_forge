from unittest import mock
from magnumclient.common import cliutils
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1 import certificates_shell
@mock.patch('os.path.isfile')
@mock.patch('magnumclient.v1.clusters.ClusterManager.get')
@mock.patch('magnumclient.v1.certificates.CertificateManager.create')
def test_cluster_ca_sign_with_not_csr(self, mock_cert_create, mock_cluster_get, mock_isfile):
    mock_isfile.return_value = False
    mockcluster = mock.MagicMock()
    mockcluster.status = 'CREATE_COMPLETE'
    mock_cluster_get.return_value = mockcluster
    fake_csr = 'fake-csr'
    mock_file = mock.mock_open(read_data=fake_csr)
    with mock.patch.object(certificates_shell, 'open', mock_file):
        self._test_arg_success('ca-sign --csr path/csr.pem --cluster xxx')
        mock_isfile.assert_called_once_with('path/csr.pem')
        mock_file.assert_not_called()
        mock_cert_create.assert_not_called()