import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
def set_ignored_exceptions(self, ignored_exceptions):
    """Install an exception handler for the server."""
    self.server.set_ignored_exceptions(self._server_thread, ignored_exceptions)