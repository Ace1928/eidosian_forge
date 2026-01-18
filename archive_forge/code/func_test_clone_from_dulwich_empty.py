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
def test_clone_from_dulwich_empty(self):
    old_repo_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, old_repo_dir)
    self._old_repo = Repo.init_bare(old_repo_dir)
    port = self._start_server(self._old_repo)
    new_repo_base_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, new_repo_base_dir)
    new_repo_dir = os.path.join(new_repo_base_dir, 'empty_new')
    run_git_or_fail(['clone', self.url(port), new_repo_dir], cwd=new_repo_base_dir)
    new_repo = Repo(new_repo_dir)
    self.assertReposEqual(self._old_repo, new_repo)