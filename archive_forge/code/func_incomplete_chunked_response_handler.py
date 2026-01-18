import pytest
import threading
import requests
from tests.testserver.server import Server, consume_socket_content
from .utils import override_environ
def incomplete_chunked_response_handler(sock):
    request_content = consume_socket_content(sock, timeout=0.5)
    sock.send(b'HTTP/1.1 200 OK\r\n' + b'Transfer-Encoding: chunked\r\n')
    return request_content