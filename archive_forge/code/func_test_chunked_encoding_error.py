import pytest
import threading
import requests
from tests.testserver.server import Server, consume_socket_content
from .utils import override_environ
def test_chunked_encoding_error():
    """get a ChunkedEncodingError if the server returns a bad response"""

    def incomplete_chunked_response_handler(sock):
        request_content = consume_socket_content(sock, timeout=0.5)
        sock.send(b'HTTP/1.1 200 OK\r\n' + b'Transfer-Encoding: chunked\r\n')
        return request_content
    close_server = threading.Event()
    server = Server(incomplete_chunked_response_handler)
    with server as (host, port):
        url = 'http://{}:{}/'.format(host, port)
        with pytest.raises(requests.exceptions.ChunkedEncodingError):
            r = requests.get(url)
        close_server.set()