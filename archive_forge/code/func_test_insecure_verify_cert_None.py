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
def test_insecure_verify_cert_None(self, mock_request):
    client = http.HTTPClient('https://foo', insecure=True)
    self.assertFalse(client.verify_cert)