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
def test_clone_from_root(self):
    """At the root, cloning to a simple dir should just do string append."""
    orig_transport = self.get_transport()
    root_transport = orig_transport.clone('/')
    self.assertEqual(root_transport.base + '.bzr/', root_transport.clone('.bzr').base)