import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
def pending_exception(self):
    """Raise uncaught exception in the server."""
    self.server._pending_exception(self._server_thread)