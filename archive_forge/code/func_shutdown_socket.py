import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
def shutdown_socket(self, sock):
    """Properly shutdown a socket.

        This should be called only when no other thread is trying to use the
        socket.
        """
    try:
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()
    except Exception as e:
        if self.ignored_exceptions(e):
            pass
        else:
            raise