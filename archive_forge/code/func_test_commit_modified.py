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
def test_commit_modified(self):
    r = self._repo
    with open(os.path.join(r.path, 'a'), 'wb') as f:
        f.write(b'new contents')
    r.stage(['a'])
    commit_sha = r.do_commit(b'modified a', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0)
    self.assertEqual([self._root_commit], r[commit_sha].parents)
    a_mode, a_id = tree_lookup_path(r.get_object, r[commit_sha].tree, b'a')
    self.assertEqual(stat.S_IFREG | 420, a_mode)
    self.assertEqual(b'new contents', r[a_id].data)