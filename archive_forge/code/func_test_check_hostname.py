from tornado.concurrent import Future
from tornado import gen
from tornado import netutil
from tornado.ioloop import IOLoop
from tornado.iostream import (
from tornado.httputil import HTTPHeaders
from tornado.locks import Condition, Event
from tornado.log import gen_log
from tornado.netutil import ssl_options_to_context, ssl_wrap_socket
from tornado.platform.asyncio import AddThreadSelectorEventLoop
from tornado.tcpserver import TCPServer
from tornado.testing import (
from tornado.test.util import (
from tornado.web import RequestHandler, Application
import asyncio
import errno
import hashlib
import logging
import os
import platform
import random
import socket
import ssl
import typing
from unittest import mock
import unittest
@gen_test
def test_check_hostname(self):
    server_future = self.server_start_tls(_server_ssl_options())
    with ExpectLog(gen_log, 'SSL Error'):
        client_future = self.client_start_tls(ssl.create_default_context(), server_hostname='127.0.0.1')
        with self.assertRaises(ssl.SSLError):
            yield client_future
        with self.assertRaises(Exception):
            yield server_future