import stat
from io import BytesIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import ZERO_SHA, Blob, Commit, Tree
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_commit_handler(self):
    from fastimport import commands
    cmd = commands.CommitCommand(b'refs/heads/foo', b'mrkr', (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), b'FOO', None, [], [])
    self.processor.commit_handler(cmd)
    commit = self.repo[self.processor.last_commit]
    self.assertEqual(b'Jelmer <jelmer@samba.org>', commit.author)
    self.assertEqual(b'Jelmer <jelmer@samba.org>', commit.committer)
    self.assertEqual(b'FOO', commit.message)
    self.assertEqual([], commit.parents)
    self.assertEqual(432432432.0, commit.commit_time)
    self.assertEqual(432432432.0, commit.author_time)
    self.assertEqual(3600, commit.commit_timezone)
    self.assertEqual(3600, commit.author_timezone)
    self.assertEqual(commit, self.repo[b'refs/heads/foo'])