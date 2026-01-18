import stat
from io import BytesIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import ZERO_SHA, Blob, Commit, Tree
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_file_add(self):
    from fastimport import commands
    cmd = commands.BlobCommand(b'23', b'data')
    self.processor.blob_handler(cmd)
    cmd = commands.CommitCommand(b'refs/heads/foo', b'mrkr', (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), b'FOO', None, [], [commands.FileModifyCommand(b'path', 33188, b':23', None)])
    self.processor.commit_handler(cmd)
    commit = self.repo[self.processor.last_commit]
    self.assertEqual([(b'path', 33188, b'6320cd248dd8aeaab759d5871f8781b5c0505172')], self.repo[commit.tree].items())