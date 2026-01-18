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
def test_ensure_base_missing(self):
    """.ensure_base() should create the directory if it doesn't exist"""
    t = self.get_transport()
    t_a = t.clone('a')
    self.assertFalse(t.ensure_base())
    if t_a.is_readonly():
        self.assertRaises(TransportNotPossible, t_a.ensure_base)
        return
    self.assertTrue(t_a.ensure_base())
    self.assertTrue(t.has('a'))