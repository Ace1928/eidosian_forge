import stat
from io import BytesIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import ZERO_SHA, Blob, Commit, Tree
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_file_deleteall(self):
    from fastimport import commands
    self.simple_commit()
    commit = self.make_file_commit([commands.FileDeleteAllCommand()])
    self.assertEqual([], self.repo[commit.tree].items())