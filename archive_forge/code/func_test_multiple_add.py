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
def test_multiple_add(self):
    """Multiple add with same ID should raise a BzrError"""
    store = self.get_store()
    self.fill_store(store)
    self.assertRaises(BzrError, store.add, BytesIO(b'goodbye'), b'123123')