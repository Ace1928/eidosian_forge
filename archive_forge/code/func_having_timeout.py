import errno
import os
import re
import socket
import ssl
from contextlib import contextmanager
from ssl import SSLError
from struct import pack, unpack
from .exceptions import UnexpectedFrame
from .platform import KNOWN_TCP_OPTS, SOL_TCP
from .utils import set_cloexec
@contextmanager
def having_timeout(self, timeout):
    if timeout is None:
        yield self.sock
    else:
        sock = self.sock
        prev = sock.gettimeout()
        if prev != timeout:
            sock.settimeout(timeout)
        try:
            yield self.sock
        except SSLError as exc:
            if 'timed out' in str(exc):
                raise socket.timeout()
            elif 'The operation did not complete' in str(exc):
                raise socket.timeout()
            raise
        except OSError as exc:
            if exc.errno == errno.EWOULDBLOCK:
                raise socket.timeout()
            raise
        finally:
            if timeout != prev:
                sock.settimeout(prev)