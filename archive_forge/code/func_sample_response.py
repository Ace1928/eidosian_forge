import platform
import select
import socket
import ssl
import sys
import mock
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import SocketDummyServerTestCase, consume_socket
from urllib3.util import ssl_
from urllib3.util.ssltransport import SSLTransport
def sample_response(binary=True):
    response = b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n'
    return response if binary else response.decode('utf-8')