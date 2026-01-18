import pytest
import threading
import requests
from tests.testserver.server import Server, consume_socket_content
from .utils import override_environ
def echo_response_handler(sock):
    """Simple handler that will take request and echo it back to requester."""
    request_content = consume_socket_content(sock, timeout=0.5)
    text_200 = b'HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n%s' % (len(request_content), request_content)
    sock.send(text_200)