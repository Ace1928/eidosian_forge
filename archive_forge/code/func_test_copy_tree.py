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
def test_copy_tree(self):
    transport = self.get_transport()
    if not transport.listable():
        self.assertRaises(TransportNotPossible, transport.iter_files_recursive)
        return
    if transport.is_readonly():
        return
    self.build_tree(['from/', 'from/dir/', 'from/dir/foo', 'from/dir/bar', 'from/dir/b%25z', 'from/bar'], transport=transport)
    transport.copy_tree('from', 'to')
    paths = set(transport.iter_files_recursive())
    self.assertEqual(paths, {'from/dir/foo', 'from/dir/bar', 'from/dir/b%2525z', 'from/bar', 'to/dir/foo', 'to/dir/bar', 'to/dir/b%2525z', 'to/bar'})