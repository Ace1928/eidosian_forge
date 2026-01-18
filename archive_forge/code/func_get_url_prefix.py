import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
def get_url_prefix(self):
    """What URL prefix does this decorator produce?"""
    return self.get_decorator_class()._get_url_prefix()