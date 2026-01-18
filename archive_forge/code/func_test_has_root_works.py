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
def test_has_root_works(self):
    if self.transport_server is test_server.SmartTCPServer_for_testing:
        raise TestNotApplicable('SmartTCPServer_for_testing intentionally does not allow access to /.')
    current_transport = self.get_transport()
    self.assertTrue(current_transport.has('/'))
    root = current_transport.clone('/')
    self.assertTrue(root.has(''))