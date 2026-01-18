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
def test_insecure_connection_unusual_port(self):
    """
        Test that the connection will allow unusual ports and insecure
        schemes
        """
    conn = Connection(secure=False, host='localhost', port=8081)
    conn.connect()
    self.assertEqual(conn.connection.host, 'http://localhost:8081')
    conn2 = Connection(url='http://localhost:8081')
    conn2.connect()
    self.assertEqual(conn2.connection.host, 'http://localhost:8081')