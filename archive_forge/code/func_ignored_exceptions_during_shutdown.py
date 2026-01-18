import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
def ignored_exceptions_during_shutdown(self, e):
    if sys.platform == 'win32':
        accepted_errnos = [errno.EBADF, errno.EPIPE, errno.WSAEBADF, errno.WSAENOTSOCK, errno.WSAECONNRESET, errno.WSAENOTCONN, errno.WSAESHUTDOWN]
    else:
        accepted_errnos = [errno.EBADF, errno.ECONNRESET, errno.ENOTCONN, errno.EPIPE]
    if isinstance(e, socket.error) and e.errno in accepted_errnos:
        return True
    return False