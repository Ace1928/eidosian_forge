import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
def make_pf_transport(self, filter_func=None):
    """Make a PathFilteringTransport backed by a MemoryTransport.

        :param filter_func: by default this will be a no-op function.  Use this
            parameter to override it."""
    if filter_func is None:

        def filter_func(x):
            return x
    server = pathfilter.PathFilteringServer(transport.get_transport_from_url('memory:///foo/bar/'), filter_func)
    server.start_server()
    self.addCleanup(server.stop_server)
    return transport.get_transport_from_url(server.get_url())