import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch('urllib3.connection.HTTPSConnection')
def test_create_connection_https(self, https_conn):
    conn = mock.Mock()
    https_conn.return_value = conn
    handle = rw_handles.FileHandle(None)
    ret = handle._create_connection('https://localhost/foo?q=bar', 'GET')
    self.assertEqual(conn, ret)
    ca_store = requests.certs.where()
    conn.set_cert.assert_called_once_with(ca_certs=ca_store, cert_reqs=ssl.CERT_NONE, assert_fingerprint=None)
    conn.putrequest.assert_called_once_with('GET', '/foo?q=bar')