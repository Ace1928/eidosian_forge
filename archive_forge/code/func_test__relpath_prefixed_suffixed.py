import gzip
import os
from io import BytesIO
from ... import errors as errors
from ... import transactions, transport
from ...bzr.weave import WeaveFile
from ...errors import BzrError
from ...tests import TestCase, TestCaseInTempDir, TestCaseWithTransport
from ...transport.memory import MemoryTransport
from .store import TransportStore
from .store.text import TextStore
from .store.versioned import VersionedFileStore
def test__relpath_prefixed_suffixed(self):
    my_store = TransportStore(MockTransport(), True)
    my_store.register_suffix('bar')
    my_store.register_suffix('baz')
    self.assertEqual('45/foo.baz', my_store._relpath(b'foo', ['baz']))
    self.assertEqual('45/foo.bar.baz', my_store._relpath(b'foo', ['bar', 'baz']))