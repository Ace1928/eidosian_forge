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
@unittest.skipIf(no_internet(), 'Internet is not reachable')
def test_connection_timeout_raised(self):
    """
        Test that the connection times out
        """
    conn = LibcloudConnection(host='localhost', port=8080, timeout=0.1)
    host = 'http://10.255.255.1'
    with self.assertRaises(ConnectTimeout):
        conn.request('GET', host)