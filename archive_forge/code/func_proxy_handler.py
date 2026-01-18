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
def proxy_handler(listener):
    sock = listener.accept()[0]
    with self.server_context.wrap_socket(sock, server_side=True) as client_sock:
        upstream_sock = socket.create_connection((self.destination_server_host, self.destination_server_port))
        self._read_write_loop(client_sock, upstream_sock)
        upstream_sock.close()
        client_sock.close()