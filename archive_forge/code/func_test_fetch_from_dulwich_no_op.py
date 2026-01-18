import errno
import os
import shutil
import socket
import tempfile
from ...objects import hex_to_sha
from ...protocol import CAPABILITY_SIDE_BAND_64K
from ...repo import Repo
from ...server import ReceivePackHandler
from ..utils import tear_down_repo
from .utils import require_git_version, run_git_or_fail
def test_fetch_from_dulwich_no_op(self):
    self._old_repo = self.import_repo('server_old.export')
    self._new_repo = self.import_repo('server_old.export')
    self.assertReposEqual(self._old_repo, self._new_repo)
    port = self._start_server(self._new_repo)
    run_git_or_fail(['fetch', self.url(port), *self.branch_args()], cwd=self._old_repo.path)
    self._old_repo.object_store._pack_cache_time = 0
    self.assertReposEqual(self._old_repo, self._new_repo)