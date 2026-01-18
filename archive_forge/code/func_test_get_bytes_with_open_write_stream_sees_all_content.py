import os
import stat
import sys
from io import BytesIO
from .. import errors, osutils, pyutils, tests
from .. import transport as _mod_transport
from .. import urlutils
from ..errors import ConnectionError, PathError, TransportNotPossible
from ..osutils import getcwd
from ..transport import (ConnectedTransport, FileExists, NoSuchFile, Transport,
from ..transport.memory import MemoryTransport
from ..transport.remote import RemoteTransport
from . import TestNotApplicable, TestSkipped, multiply_tests, test_server
from .test_transport import TestTransportImplementation
def test_get_bytes_with_open_write_stream_sees_all_content(self):
    t = self.get_transport()
    if t.is_readonly():
        return
    with t.open_write_stream('foo') as handle:
        handle.write(b'b')
        self.assertEqual(b'b', t.get_bytes('foo'))
        with t.get('foo') as f:
            self.assertEqual(b'b', f.read())