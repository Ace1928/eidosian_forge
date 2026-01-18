import collections
import sys
import warnings
from . import protocols
from . import transports
from .log import logger
def feed_appdata(self, data, offset=0):
    """Feed plaintext data into the pipe.

        Return an (ssldata, offset) tuple. The ssldata element is a list of
        buffers containing record level data that needs to be sent to the
        remote SSL instance. The offset is the number of plaintext bytes that
        were processed, which may be less than the length of data.

        NOTE: In case of short writes, this call MUST be retried with the SAME
        buffer passed into the *data* argument (i.e. the id() must be the
        same). This is an OpenSSL requirement. A further particularity is that
        a short write will always have offset == 0, because the _ssl module
        does not enable partial writes. And even though the offset is zero,
        there will still be encrypted data in ssldata.
        """
    assert 0 <= offset <= len(data)
    if self._state == _UNWRAPPED:
        if offset < len(data):
            ssldata = [data[offset:]]
        else:
            ssldata = []
        return (ssldata, len(data))
    ssldata = []
    view = memoryview(data)
    while True:
        self._need_ssldata = False
        try:
            if offset < len(view):
                offset += self._sslobj.write(view[offset:])
        except ssl.SSLError as exc:
            if exc.reason == 'PROTOCOL_IS_SHUTDOWN':
                exc.errno = ssl.SSL_ERROR_WANT_READ
            if exc.errno not in (ssl.SSL_ERROR_WANT_READ, ssl.SSL_ERROR_WANT_WRITE, ssl.SSL_ERROR_SYSCALL):
                raise
            self._need_ssldata = exc.errno == ssl.SSL_ERROR_WANT_READ
        if self._outgoing.pending:
            ssldata.append(self._outgoing.read())
        if offset == len(view) or self._need_ssldata:
            break
    return (ssldata, offset)