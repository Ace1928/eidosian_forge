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
def test_create_prefix(self):
    t = self.get_transport()
    sub = t.clone('foo').clone('bar')
    try:
        sub.create_prefix()
    except TransportNotPossible:
        self.assertTrue(t.is_readonly())
    else:
        self.assertTrue(t.has('foo/bar'))