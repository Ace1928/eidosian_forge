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
def get_populated_store(self, prefixed=False, store_class=TextStore, compressed=False):
    my_store = store_class(MemoryTransport(), prefixed, compressed=compressed)
    my_store.register_suffix('sig')
    stream = BytesIO(b'signature')
    my_store.add(stream, b'foo', 'sig')
    stream = BytesIO(b'content')
    my_store.add(stream, b'foo')
    stream = BytesIO(b'signature for missing base')
    my_store.add(stream, b'missing', 'sig')
    return my_store