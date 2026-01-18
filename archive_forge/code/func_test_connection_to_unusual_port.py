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
def test_connection_to_unusual_port(self):
    conn = LibcloudConnection(host='localhost', port=8080)
    self.assertIsNone(conn.proxy_scheme)
    self.assertIsNone(conn.proxy_host)
    self.assertIsNone(conn.proxy_port)
    self.assertEqual(conn.host, 'http://localhost:8080')
    conn = LibcloudConnection(host='localhost', port=80)
    self.assertEqual(conn.host, 'http://localhost')