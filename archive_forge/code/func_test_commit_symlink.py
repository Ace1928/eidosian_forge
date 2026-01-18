import glob
import locale
import os
import shutil
import stat
import sys
import tempfile
import warnings
from dulwich import errors, objects, porcelain
from dulwich.tests import TestCase, skipIf
from ..config import Config
from ..errors import NotGitRepository
from ..object_store import tree_lookup_path
from ..repo import (
from .utils import open_repo, setup_warning_catcher, tear_down_repo
import sys
from dulwich.repo import Repo
@skipIf(not getattr(os, 'symlink', None), 'Requires symlink support')
def test_commit_symlink(self):
    r = self._repo
    os.symlink('a', os.path.join(r.path, 'b'))
    r.stage(['a', 'b'])
    commit_sha = r.do_commit(b'Symlink b', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0)
    self.assertEqual([self._root_commit], r[commit_sha].parents)
    b_mode, b_id = tree_lookup_path(r.get_object, r[commit_sha].tree, b'b')
    self.assertTrue(stat.S_ISLNK(b_mode))
    self.assertEqual(b'a', r[b_id].data)