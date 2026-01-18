import stat
from io import BytesIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import ZERO_SHA, Blob, Commit, Tree
from ..repo import MemoryRepo
from .utils import build_commit_graph
def make_file_commit(self, file_cmds):
    """Create a trivial commit with the specified file commands.

        Args:
          file_cmds: File commands to run.
        Returns: The created commit object
        """
    from fastimport import commands
    cmd = commands.CommitCommand(b'refs/heads/foo', b'mrkr', (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), (b'Jelmer', b'jelmer@samba.org', 432432432.0, 3600), b'FOO', None, [], file_cmds)
    self.processor.commit_handler(cmd)
    return self.repo[self.processor.last_commit]