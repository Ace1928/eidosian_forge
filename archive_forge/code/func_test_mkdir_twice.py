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
def test_mkdir_twice(self):
    t = memory.MemoryTransport()
    t.mkdir('dir')
    self.assertRaises(FileExists, t.mkdir, 'dir')