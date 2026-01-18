import socket
from unittest import mock
import io
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
import testtools
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
from heatclient.tests.unit import fakes
def test_get_system_ca_file(self, mock_request):
    chosen = '/etc/ssl/certs/ca-certificates.crt'
    with mock.patch('os.path.exists') as mock_os:
        mock_os.return_value = chosen
        ca = http.get_system_ca_file()
        self.assertEqual(chosen, ca)
        mock_os.assert_called_once_with(chosen)