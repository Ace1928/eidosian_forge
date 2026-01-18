import struct
import tarfile
from io import BytesIO
from unittest import skipUnless
from dulwich.tests import TestCase
from ..archive import tar_stream
from ..object_store import MemoryObjectStore
from ..objects import Blob, Tree
from .utils import build_commit_graph
def test_gzip_mtime(self):
    stream = self._get_example_tar_stream(mtime=1234, format='gz')
    expected_mtime = struct.pack('<L', 1234)
    self.assertEqual(stream.getvalue()[4:8], expected_mtime)