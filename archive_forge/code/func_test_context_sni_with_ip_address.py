from test import notPyPy2
import mock
import pytest
from urllib3.exceptions import SNIMissingWarning
from urllib3.util import ssl_
@pytest.mark.parametrize(['has_sni', 'server_hostname', 'uses_sni'], [(True, '127.0.0.1', False), (False, 'www.python.org', False), (False, '0.0.0.0', False), (True, 'www.google.com', True), (True, None, False), (False, None, False)])
def test_context_sni_with_ip_address(monkeypatch, has_sni, server_hostname, uses_sni):
    monkeypatch.setattr(ssl_, 'HAS_SNI', has_sni)
    sock = mock.Mock()
    context = mock.create_autospec(ssl_.SSLContext)
    ssl_.ssl_wrap_socket(sock, server_hostname=server_hostname, ssl_context=context)
    if uses_sni:
        context.wrap_socket.assert_called_with(sock, server_hostname=server_hostname)
    else:
        context.wrap_socket.assert_called_with(sock)