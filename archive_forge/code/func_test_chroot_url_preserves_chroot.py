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
def test_chroot_url_preserves_chroot(self):
    """Calling get_transport on a chroot transport's base should produce a
        transport with exactly the same behaviour as the original chroot
        transport.

        This is so that it is not possible to escape a chroot by doing::
            url = chroot_transport.base
            parent_url = urlutils.join(url, '..')
            new_t = transport.get_transport_from_url(parent_url)
        """
    server = chroot.ChrootServer(transport.get_transport_from_url('memory:///path/subpath'))
    self.start_server(server)
    t = transport.get_transport_from_url(server.get_url())
    new_t = transport.get_transport_from_url(t.base)
    self.assertEqual(t.server, new_t.server)
    self.assertEqual(t.base, new_t.base)