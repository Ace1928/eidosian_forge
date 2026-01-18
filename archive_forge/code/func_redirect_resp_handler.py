import pytest
import threading
import requests
from tests.testserver.server import Server, consume_socket_content
from .utils import override_environ
def redirect_resp_handler(sock):
    consume_socket_content(sock, timeout=0.5)
    location = u'//{}:{}/{}'.format(host, port, path)
    sock.send(b'HTTP/1.1 301 Moved Permanently\r\nContent-Length: 0\r\nLocation: ' + location.encode('utf8') + b'\r\n\r\n')
    redirect_request.append(consume_socket_content(sock, timeout=0.5))
    sock.send(b'HTTP/1.1 200 OK\r\n\r\n')