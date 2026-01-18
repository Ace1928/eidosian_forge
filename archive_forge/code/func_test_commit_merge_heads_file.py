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
def test_commit_merge_heads_file(self):
    tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    r = Repo.init(tmp_dir)
    with open(os.path.join(r.path, 'a'), 'w') as f:
        f.write('initial text')
    c1 = r.do_commit(b'initial commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0)
    with open(os.path.join(r.path, 'a'), 'w') as f:
        f.write('merged text')
    with open(os.path.join(r.path, '.git', 'MERGE_HEAD'), 'w') as f:
        f.write('c27a2d21dd136312d7fa9e8baabb82561a1727d0\n')
    r.stage(['a'])
    commit_sha = r.do_commit(b'deleted a', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0)
    self.assertEqual([c1, b'c27a2d21dd136312d7fa9e8baabb82561a1727d0'], r[commit_sha].parents)