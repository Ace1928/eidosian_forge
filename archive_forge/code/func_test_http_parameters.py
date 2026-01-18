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
def test_http_parameters(self):
    from breezy.tests.http_server import HttpServer
    server = HttpServer()
    self.start_server(server)
    t = self.get_nfs_transport(server.get_url())
    self.assertIsInstance(t, fakenfs.FakeNFSTransportDecorator)
    self.assertEqual(False, t.listable())
    self.assertEqual(True, t.is_readonly())