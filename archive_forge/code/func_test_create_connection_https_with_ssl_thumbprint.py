import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch('urllib3.connection.HTTPSConnection')
def test_create_connection_https_with_ssl_thumbprint(self, https_conn):
    conn = mock.Mock()
    https_conn.return_value = conn
    handle = rw_handles.FileHandle(None)
    cacerts = mock.sentinel.cacerts
    thumbprint = mock.sentinel.thumbprint
    ret = handle._create_connection('https://localhost/foo?q=bar', 'GET', cacerts=cacerts, ssl_thumbprint=thumbprint)
    self.assertEqual(conn, ret)
    conn.set_cert.assert_called_once_with(ca_certs=cacerts, cert_reqs=None, assert_fingerprint=thumbprint)