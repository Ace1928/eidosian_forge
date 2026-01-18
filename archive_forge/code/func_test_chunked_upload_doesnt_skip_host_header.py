import pytest
import threading
import requests
from tests.testserver.server import Server, consume_socket_content
from .utils import override_environ
def test_chunked_upload_doesnt_skip_host_header():
    """Ensure we don't omit all Host headers with chunked requests."""
    close_server = threading.Event()
    server = Server(echo_response_handler, wait_to_close_event=close_server)
    data = iter([b'a', b'b', b'c'])
    with server as (host, port):
        expected_host = '{}:{}'.format(host, port)
        url = 'http://{}:{}/'.format(host, port)
        r = requests.post(url, data=data, stream=True)
        close_server.set()
    expected_header = b'Host: %s\r\n' % expected_host.encode('utf-8')
    assert expected_header in r.content
    assert r.content.count(b'Host: ') == 1