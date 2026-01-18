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
def test_iter_files_recursive(self):
    t = memory.MemoryTransport()
    t.mkdir('dir')
    t.put_bytes('dir/foo', b'content')
    t.put_bytes('dir/bar', b'content')
    t.put_bytes('bar', b'content')
    paths = set(t.iter_files_recursive())
    self.assertEqual({'dir/foo', 'dir/bar', 'bar'}, paths)