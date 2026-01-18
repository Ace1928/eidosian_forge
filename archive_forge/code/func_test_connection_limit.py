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
def test_connection_limit(self: typing.Any):
    with closing(self.create_client(max_clients=2)) as client:
        self.assertEqual(client.max_clients, 2)
        seen = []
        for i in range(4):

            def cb(fut, i=i):
                seen.append(i)
                self.stop()
            client.fetch(self.get_url('/trigger')).add_done_callback(cb)
        self.wait(condition=lambda: len(self.triggers) == 2)
        self.assertEqual(len(client.queue), 2)
        self.triggers.popleft()()
        self.triggers.popleft()()
        self.wait(condition=lambda: len(self.triggers) == 2 and len(seen) == 2)
        self.assertEqual(set(seen), set([0, 1]))
        self.assertEqual(len(client.queue), 0)
        self.triggers.popleft()()
        self.triggers.popleft()()
        self.wait(condition=lambda: len(seen) == 4)
        self.assertEqual(set(seen), set([0, 1, 2, 3]))
        self.assertEqual(len(self.triggers), 0)