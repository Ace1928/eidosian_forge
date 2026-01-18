import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
def process_request_thread(self, started, detached, stopped, request, client_address):
    started.set()
    detached.wait()
    socketserver.ThreadingTCPServer.process_request_thread(self, request, client_address)
    self.close_request(request)
    stopped.set()