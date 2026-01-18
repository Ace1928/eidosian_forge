import stat
from io import BytesIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import ZERO_SHA, Blob, Commit, Tree
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_import_stream(self):
    markers = self.processor.import_stream(BytesIO(b'blob\nmark :1\ndata 11\ntext for a\n\ncommit refs/heads/master\nmark :2\ncommitter Joe Foo <joe@foo.com> 1288287382 +0000\ndata 20\n<The commit message>\nM 100644 :1 a\n\n'))
    self.assertEqual(2, len(markers))
    self.assertIsInstance(self.repo[markers[b'1']], Blob)
    self.assertIsInstance(self.repo[markers[b'2']], Commit)