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
def test___iter__no_suffix(self):
    my_store = TextStore(MemoryTransport(), prefixed=False, compressed=False)
    stream = BytesIO(b'content')
    my_store.add(stream, b'foo')
    self.assertEqual({b'foo'}, set(my_store.__iter__()))