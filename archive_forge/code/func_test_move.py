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
def test_move(self):
    t = self.get_transport()
    if t.is_readonly():
        return
    t.put_bytes('a', b'a first file\n')
    self.assertEqual([True, False], [t.has(n) for n in ['a', 'b']])
    t.move('a', 'b')
    self.assertTrue(t.has('b'))
    self.assertFalse(t.has('a'))
    self.check_transport_contents(b'a first file\n', t, 'b')
    self.assertEqual([False, True], [t.has(n) for n in ['a', 'b']])
    t.put_bytes('c', b'c this file\n')
    t.move('c', 'b')
    self.assertFalse(t.has('c'))
    self.check_transport_contents(b'c this file\n', t, 'b')