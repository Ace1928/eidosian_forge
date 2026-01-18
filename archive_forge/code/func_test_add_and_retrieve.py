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
def test_add_and_retrieve(self):
    store = self.get_store()
    store.add(BytesIO(b'hello'), b'aa')
    self.assertNotEqual(store.get(b'aa'), None)
    self.assertEqual(store.get(b'aa').read(), b'hello')
    store.add(BytesIO(b'hello world'), b'bb')
    self.assertNotEqual(store.get(b'bb'), None)
    self.assertEqual(store.get(b'bb').read(), b'hello world')