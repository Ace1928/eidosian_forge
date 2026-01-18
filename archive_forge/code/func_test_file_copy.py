import stat
from io import BytesIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import ZERO_SHA, Blob, Commit, Tree
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_file_copy(self):
    from fastimport import commands
    self.simple_commit()
    commit = self.make_file_commit([commands.FileCopyCommand(b'path', b'new_path')])
    self.assertEqual([(b'new_path', 33188, b'6320cd248dd8aeaab759d5871f8781b5c0505172'), (b'path', 33188, b'6320cd248dd8aeaab759d5871f8781b5c0505172')], self.repo[commit.tree].items())