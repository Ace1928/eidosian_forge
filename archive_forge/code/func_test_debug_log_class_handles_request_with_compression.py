import os
import sys
import zlib
from io import StringIO
from unittest import mock
import requests_mock
import libcloud
from libcloud.http import LibcloudConnection
from libcloud.test import unittest
from libcloud.common.base import Connection
from libcloud.utils.loggingconnection import LoggingConnection
def test_debug_log_class_handles_request_with_compression(self):
    request = zlib.compress(b'data')
    with StringIO() as fh:
        libcloud.enable_debug(fh)
        conn = Connection(url='http://test.com/')
        conn.connect()
        self.assertEqual(conn.connection.host, 'http://test.com')
        with requests_mock.mock() as m:
            m.get('http://test.com/test', content=request, headers={'content-encoding': 'zlib'})
            conn.request('/test')
        log = fh.getvalue()
    self.assertTrue(isinstance(conn.connection, LoggingConnection))
    self.assertIn('-i -X GET', log)