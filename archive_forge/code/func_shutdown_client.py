import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
def shutdown_client(self, client):
    sock, addr, connection_thread = client
    self.shutdown_socket(sock)
    if connection_thread is not None:
        if debug_threads():
            sys.stderr.write('Client thread %s will be joined\n' % (connection_thread.name,))
        connection_thread.join()