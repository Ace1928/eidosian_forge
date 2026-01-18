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
@pytest.fixture(scope='function')
def requires_tlsv1_2(supported_tls_versions):
    """Test requires TLSv1.2 available"""
    if not hasattr(ssl, 'PROTOCOL_TLSv1_2') or 'TLSv1.2' not in supported_tls_versions:
        pytest.skip('Test requires TLSv1.2')