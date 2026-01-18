import collections
import contextlib
import platform
import socket
import ssl
import sys
import threading
import pytest
import trustme
from tornado import ioloop, web
from dummyserver.handlers import TestingApp
from dummyserver.proxy import ProxyHandler
from dummyserver.server import HAS_IPV6, run_tornado_app
from dummyserver.testcase import HTTPSDummyServerTestCase
from urllib3.util import ssl_
from .tz_stub import stub_timezone_ctx
@pytest.fixture
def no_san_server(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp('certs')
    ca = trustme.CA()
    server_cert = ca.issue_cert(common_name=u'localhost')
    with run_server_in_thread('https', 'localhost', tmpdir, ca, server_cert) as cfg:
        yield cfg