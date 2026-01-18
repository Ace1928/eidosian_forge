import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch('urllib3.connection.HTTPConnection')
def test_create_connection_http(self, http_conn):
    conn = mock.Mock()
    http_conn.return_value = conn
    handle = rw_handles.FileHandle(None)
    ret = handle._create_connection('http://localhost/foo?q=bar', 'GET')
    self.assertEqual(conn, ret)
    conn.putrequest.assert_called_once_with('GET', '/foo?q=bar')