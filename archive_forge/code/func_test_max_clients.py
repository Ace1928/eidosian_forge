import collections
from contextlib import closing
import errno
import logging
import os
import re
import socket
import ssl
import sys
import typing  # noqa: F401
from tornado.escape import to_unicode, utf8
from tornado import gen, version
from tornado.httpclient import AsyncHTTPClient
from tornado.httputil import HTTPHeaders, ResponseStartLine
from tornado.ioloop import IOLoop
from tornado.iostream import UnsatisfiableReadError
from tornado.locks import Event
from tornado.log import gen_log
from tornado.netutil import Resolver, bind_sockets
from tornado.simple_httpclient import (
from tornado.test.httpclient_test import (
from tornado.test import httpclient_test
from tornado.testing import (
from tornado.test.util import skipOnTravis, skipIfNoIPv6, refusing_port
from tornado.web import RequestHandler, Application, url, stream_request_body
def test_max_clients(self):
    AsyncHTTPClient.configure(SimpleAsyncHTTPClient)
    with closing(AsyncHTTPClient(force_instance=True)) as client:
        self.assertEqual(client.max_clients, 10)
    with closing(AsyncHTTPClient(max_clients=11, force_instance=True)) as client:
        self.assertEqual(client.max_clients, 11)
    AsyncHTTPClient.configure(SimpleAsyncHTTPClient, max_clients=12)
    with closing(AsyncHTTPClient(force_instance=True)) as client:
        self.assertEqual(client.max_clients, 12)
    with closing(AsyncHTTPClient(max_clients=13, force_instance=True)) as client:
        self.assertEqual(client.max_clients, 13)
    with closing(AsyncHTTPClient(max_clients=14, force_instance=True)) as client:
        self.assertEqual(client.max_clients, 14)