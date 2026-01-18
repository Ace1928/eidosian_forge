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
@contextlib.contextmanager
def run_server_in_thread(scheme, host, tmpdir, ca, server_cert):
    ca_cert_path = str(tmpdir / 'ca.pem')
    ca.cert_pem.write_to_path(ca_cert_path)
    server_certs = _write_cert_to_dir(server_cert, tmpdir)
    io_loop = ioloop.IOLoop.current()
    app = web.Application([('.*', TestingApp)])
    server, port = run_tornado_app(app, io_loop, server_certs, scheme, host)
    server_thread = threading.Thread(target=io_loop.start)
    server_thread.start()
    yield ServerConfig(host, port, ca_cert_path)
    io_loop.add_callback(server.stop)
    io_loop.add_callback(io_loop.stop)
    server_thread.join()