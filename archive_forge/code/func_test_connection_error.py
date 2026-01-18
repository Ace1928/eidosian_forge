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
def test_connection_error(self):
    """ConnectionError is raised when connection is impossible.

        The error should be raised from the first operation on the transport.
        """
    try:
        url = self._server.get_bogus_url()
    except NotImplementedError:
        raise TestSkipped('Transport %s has no bogus URL support.' % self._server.__class__)
    t = _mod_transport.get_transport_from_url(url)
    self.assertRaises((ConnectionError, NoSuchFile), t.get, '.bzr/branch')