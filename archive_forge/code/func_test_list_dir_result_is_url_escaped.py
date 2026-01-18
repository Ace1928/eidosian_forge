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
def test_list_dir_result_is_url_escaped(self):
    t = self.get_transport()
    if not t.listable():
        raise TestSkipped('transport not listable')
    if not t.is_readonly():
        self.build_tree(['a/', 'a/%'], transport=t)
    else:
        self.build_tree(['a/', 'a/%'])
    names = list(t.list_dir('a'))
    self.assertEqual(['%25'], names)
    self.assertIsInstance(names[0], str)