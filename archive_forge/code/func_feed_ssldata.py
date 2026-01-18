import collections
import sys
import warnings
from . import protocols
from . import transports
from .log import logger
def feed_ssldata(self, data, only_handshake=False):
    """Feed SSL record level data into the pipe.

        The data must be a bytes instance. It is OK to send an empty bytes
        instance. This can be used to get ssldata for a handshake initiated by
        this endpoint.

        Return a (ssldata, appdata) tuple. The ssldata element is a list of
        buffers containing SSL data that needs to be sent to the remote SSL.

        The appdata element is a list of buffers containing plaintext data that
        needs to be forwarded to the application. The appdata list may contain
        an empty buffer indicating an SSL "close_notify" alert. This alert must
        be acknowledged by calling shutdown().
        """
    if self._state == _UNWRAPPED:
        if data:
            appdata = [data]
        else:
            appdata = []
        return ([], appdata)
    self._need_ssldata = False
    if data:
        self._incoming.write(data)
    ssldata = []
    appdata = []
    try:
        if self._state == _DO_HANDSHAKE:
            self._sslobj.do_handshake()
            self._state = _WRAPPED
            if self._handshake_cb:
                self._handshake_cb(None)
            if only_handshake:
                return (ssldata, appdata)
        if self._state == _WRAPPED:
            while True:
                chunk = self._sslobj.read(self.max_size)
                appdata.append(chunk)
                if not chunk:
                    break
        elif self._state == _SHUTDOWN:
            self._sslobj.unwrap()
            self._sslobj = None
            self._state = _UNWRAPPED
            if self._shutdown_cb:
                self._shutdown_cb()
        elif self._state == _UNWRAPPED:
            appdata.append(self._incoming.read())
    except (ssl.SSLError, ssl.CertificateError) as exc:
        if getattr(exc, 'errno', None) not in (ssl.SSL_ERROR_WANT_READ, ssl.SSL_ERROR_WANT_WRITE, ssl.SSL_ERROR_SYSCALL):
            if self._state == _DO_HANDSHAKE and self._handshake_cb:
                self._handshake_cb(exc)
            raise
        self._need_ssldata = exc.errno == ssl.SSL_ERROR_WANT_READ
    if self._outgoing.pending:
        ssldata.append(self._outgoing.read())
    return (ssldata, appdata)