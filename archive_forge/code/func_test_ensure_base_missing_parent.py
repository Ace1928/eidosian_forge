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
def test_ensure_base_missing_parent(self):
    """.ensure_base() will fail if the parent dir doesn't exist"""
    t = self.get_transport()
    if t.is_readonly():
        return
    t_a = t.clone('a')
    t_b = t_a.clone('b')
    self.assertRaises(NoSuchFile, t_b.ensure_base)