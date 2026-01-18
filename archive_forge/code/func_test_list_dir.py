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
def test_list_dir(self):
    t = memory.MemoryTransport()
    t.put_bytes('foo', b'content')
    t.mkdir('dir')
    t.put_bytes('dir/subfoo', b'content')
    t.put_bytes('dirlike', b'content')
    self.assertEqual(['dir', 'dirlike', 'foo'], sorted(t.list_dir('.')))
    self.assertEqual(['subfoo'], sorted(t.list_dir('dir')))