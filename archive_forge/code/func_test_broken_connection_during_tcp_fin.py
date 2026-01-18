import errno
import socket
import time
import logging
import traceback as traceback_
from collections import namedtuple
import http.client
import urllib.request
import pytest
from jaraco.text import trim, unwrap
from cheroot.test import helper, webtest
from cheroot._compat import IS_CI, IS_MACOS, IS_PYPY, IS_WINDOWS
import cheroot.server
@pytest.mark.parametrize(('simulated_exception', 'error_number', 'exception_leaks'), (pytest.param(socket.error, errno.ECONNRESET, False, id='socket.error(ECONNRESET)'), pytest.param(socket.error, errno.EPIPE, False, id='socket.error(EPIPE)'), pytest.param(socket.error, errno.ENOTCONN, False, id='simulated socket.error(ENOTCONN)'), pytest.param(None, errno.ENOTCONN, False, id='real socket.error(ENOTCONN)', marks=pytest.mark.xfail(IS_WINDOWS, reason='Now reproducible this way on Windows')), pytest.param(socket.error, errno.ESHUTDOWN, False, id='socket.error(ESHUTDOWN)'), pytest.param(RuntimeError, 666, True, id='RuntimeError(666)'), pytest.param(socket.error, -1, True, id='socket.error(-1)')) + (pytest.param(ConnectionResetError, errno.ECONNRESET, False, id='ConnectionResetError(ECONNRESET)'), pytest.param(BrokenPipeError, errno.EPIPE, False, id='BrokenPipeError(EPIPE)'), pytest.param(BrokenPipeError, errno.ESHUTDOWN, False, id='BrokenPipeError(ESHUTDOWN)')))
def test_broken_connection_during_tcp_fin(error_number, exception_leaks, mocker, monkeypatch, simulated_exception, test_client):
    """Test there's no traceback on broken connection during close.

    It artificially causes :py:data:`~errno.ECONNRESET` /
    :py:data:`~errno.EPIPE` / :py:data:`~errno.ESHUTDOWN` /
    :py:data:`~errno.ENOTCONN` as well as unrelated :py:exc:`RuntimeError`
    and :py:exc:`socket.error(-1) <socket.error>` on the server socket when
    :py:meth:`socket.shutdown() <socket.socket.shutdown>` is called. It's
    triggered by closing the client socket before the server had a chance
    to respond.

    The expectation is that only :py:exc:`RuntimeError` and a
    :py:exc:`socket.error` with an unusual error code would leak.

    With the :py:data:`None`-parameter, a real non-simulated
    :py:exc:`OSError(107, 'Transport endpoint is not connected')
    <OSError>` happens.
    """
    exc_instance = None if simulated_exception is None else simulated_exception(error_number, 'Simulated socket error')
    old_close_kernel_socket = test_client.server_instance.ConnectionClass._close_kernel_socket

    def _close_kernel_socket(self):
        monkeypatch.setattr(self, 'socket', mocker.mock_module.Mock(wraps=self.socket))
        if exc_instance is not None:
            monkeypatch.setattr(self.socket, 'shutdown', mocker.mock_module.Mock(side_effect=exc_instance))
        _close_kernel_socket.fin_spy = mocker.spy(self.socket, 'shutdown')
        try:
            old_close_kernel_socket(self)
        except simulated_exception:
            _close_kernel_socket.exception_leaked = True
        else:
            _close_kernel_socket.exception_leaked = False
    monkeypatch.setattr(test_client.server_instance.ConnectionClass, '_close_kernel_socket', _close_kernel_socket)
    conn = test_client.get_connection()
    conn.auto_open = False
    conn.connect()
    conn.send(b'GET /hello HTTP/1.1')
    conn.send(('Host: %s' % conn.host).encode('ascii'))
    conn.close()
    for _ in range(10 * (2 if IS_SLOW_ENV else 1)):
        time.sleep(0.1)
        if hasattr(_close_kernel_socket, 'exception_leaked'):
            break
    if exc_instance is not None:
        assert _close_kernel_socket.fin_spy.spy_exception is exc_instance
    else:
        assert isinstance(_close_kernel_socket.fin_spy.spy_exception, socket.error)
        assert _close_kernel_socket.fin_spy.spy_exception.errno == error_number
    assert _close_kernel_socket.exception_leaked is exception_leaks