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
def test_fakenfs_server_default(self):
    server = test_server.FakeNFSServer()
    self.start_server(server)
    self.assertStartsWith(server.get_url(), 'fakenfs+')
    t = transport.get_transport_from_url(server.get_url())
    self.assertIsInstance(t, fakenfs.FakeNFSTransportDecorator)