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
def test_local_fdatasync_calls_fdatasync(self):
    """Check fdatasync on a stream tries to flush the data to the OS.

        We can't easily observe the external effect but we can at least see
        it's called.
        """
    sentinel = object()
    fdatasync = getattr(os, 'fdatasync', sentinel)
    if fdatasync is sentinel:
        raise tests.TestNotApplicable('fdatasync not supported')
    t = self.get_transport('.')
    calls = self.recordCalls(os, 'fdatasync')
    w = t.open_write_stream('out')
    w.write(b'foo')
    w.fdatasync()
    with open('out', 'rb') as f:
        self.assertEqual(f.read(), b'foo')
    self.assertEqual(len(calls), 1, calls)