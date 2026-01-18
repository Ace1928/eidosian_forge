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
def test_abspath_of_root(self):
    t = memory.MemoryTransport()
    self.assertEqual('memory:///', t.base)
    self.assertEqual('memory:///', t.abspath('/'))