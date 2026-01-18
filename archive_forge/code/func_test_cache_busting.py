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
def test_cache_busting(self):
    params1 = {'foo1': 'bar1', 'foo2': 'bar2'}
    params2 = [('foo1', 'bar1'), ('foo2', 'bar2')]
    con = Connection()
    con.connection = Mock()
    con.pre_connect_hook = Mock()
    con.pre_connect_hook.return_value = ({}, {})
    con.cache_busting = False
    con.request(action='/path', params=params1)
    args, kwargs = con.pre_connect_hook.call_args
    self.assertFalse('cache-busting' in args[0])
    self.assertEqual(args[0], params1)
    con.request(action='/path', params=params2)
    args, kwargs = con.pre_connect_hook.call_args
    self.assertFalse('cache-busting' in args[0])
    self.assertEqual(args[0], params2)
    con.cache_busting = True
    con.request(action='/path', params=params1)
    args, kwargs = con.pre_connect_hook.call_args
    self.assertTrue('cache-busting' in args[0])
    con.request(action='/path', params=params2)
    args, kwargs = con.pre_connect_hook.call_args
    self.assertTrue('cache-busting' in args[0][len(params2)])