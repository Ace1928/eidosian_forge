from concurrent.futures.thread import ThreadPoolExecutor
from traceback import print_tb
import pytest
import portend
import requests
from requests_toolbelt.sessions import BaseUrlSession as Session
from jaraco.context import ExceptionTrap
from cheroot import wsgi
from cheroot._compat import IS_MACOS, IS_WINDOWS
def test_gateway_start_response_called_twice(monkeypatch):
    """Verify that repeat calls of ``Gateway.start_response()`` fail."""
    monkeypatch.setattr(wsgi.Gateway, 'get_environ', lambda self: {})
    wsgi_gateway = wsgi.Gateway(None)
    wsgi_gateway.started_response = True
    err_msg = '^WSGI start_response called a second time with no exc_info.$'
    with pytest.raises(RuntimeError, match=err_msg):
        wsgi_gateway.start_response('200', (), None)