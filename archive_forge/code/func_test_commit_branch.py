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
def test_commit_branch(self):
    r = self._repo
    commit_sha = r.do_commit(b'commit to branch', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0, ref=b'refs/heads/new_branch')
    self.assertEqual(self._root_commit, r[b'HEAD'].id)
    self.assertEqual(commit_sha, r[b'refs/heads/new_branch'].id)
    self.assertEqual([], r[commit_sha].parents)
    self.assertIn(b'refs/heads/new_branch', r)
    new_branch_head = commit_sha
    commit_sha = r.do_commit(b'commit to branch 2', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0, ref=b'refs/heads/new_branch')
    self.assertEqual(self._root_commit, r[b'HEAD'].id)
    self.assertEqual(commit_sha, r[b'refs/heads/new_branch'].id)
    self.assertEqual([new_branch_head], r[commit_sha].parents)