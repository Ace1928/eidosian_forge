import os
import ssl
import sys
import socket
from unittest import mock
from unittest.mock import Mock, patch
import requests_mock
from requests.exceptions import ConnectTimeout
import libcloud.common.base
from libcloud.http import LibcloudConnection, SignedHTTPSAdapter, LibcloudBaseConnection
from libcloud.test import unittest, no_internet
from libcloud.utils.py3 import assertRaisesRegex
from libcloud.common.base import Response, Connection, CertificateConnection
from libcloud.utils.retry import RETRY_EXCEPTIONS, Retry, RetryForeverOnRateLimitError
from libcloud.common.exceptions import RateLimitReachedError
def test_connect_with_prefix(self):
    """
        Test that a connection with a base path (e.g. /v1/) will
        add the base path to requests
        """
    conn = Connection(url='http://test.com/')
    conn.connect()
    conn.request_path = '/v1'
    self.assertEqual(conn.connection.host, 'http://test.com')
    with requests_mock.mock() as m:
        m.get('http://test.com/v1/test', text='data')
        response = conn.request('/test')
    self.assertEqual(response.body, 'data')