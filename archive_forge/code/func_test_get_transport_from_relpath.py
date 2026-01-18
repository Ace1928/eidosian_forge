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
def test_get_transport_from_relpath(self):
    t = transport.get_transport('.')
    self.assertIsInstance(t, local.LocalTransport)
    self.assertEqual(t.base, urlutils.local_path_to_url('.') + '/')