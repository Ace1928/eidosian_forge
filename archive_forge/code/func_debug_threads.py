import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
def debug_threads():
    from breezy import tests
    return 'threads' in tests.selftest_debug_flags