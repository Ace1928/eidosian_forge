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
def test_get_smart_medium(self):
    """All transports must either give a smart medium, or know they can't.
        """
    transport = self.get_transport()
    try:
        client_medium = transport.get_smart_medium()
    except errors.NoSmartMedium:
        pass
    else:
        from ..bzr.smart import medium
        self.assertIsInstance(client_medium, medium.SmartClientMedium)