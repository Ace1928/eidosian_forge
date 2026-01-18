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
def test_hook_post_connection_one(self):
    """Fire post_connect hook after a ConnectedTransport is first used"""
    log = []
    Transport.hooks.install_named_hook('post_connect', log.append, None)
    t = self.get_transport()
    self.assertEqual([], log)
    t.has('non-existant')
    if isinstance(t, RemoteTransport):
        self.assertEqual([t.get_smart_medium()], log)
    elif isinstance(t, ConnectedTransport):
        self.assertEqual([t], log)
    else:
        self.assertEqual([], log)