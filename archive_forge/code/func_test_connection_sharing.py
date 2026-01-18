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
def test_connection_sharing(self):
    t = self.get_transport()
    if not isinstance(t, ConnectedTransport):
        raise TestSkipped('not a connected transport')
    c = t.clone('subdir')
    t.has('surely_not')
    self.assertIs(t._get_connection(), c._get_connection())
    new_connection = None
    t._set_connection(new_connection)
    self.assertIs(new_connection, t._get_connection())
    self.assertIs(new_connection, c._get_connection())