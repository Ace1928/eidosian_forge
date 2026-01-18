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
def test_build_repo(self):
    r = self._repo
    self.assertEqual(b'ref: refs/heads/master', r.refs.read_ref(b'HEAD'))
    self.assertEqual(self._root_commit, r.refs[b'refs/heads/master'])
    expected_blob = objects.Blob.from_string(b'file contents')
    self.assertEqual(expected_blob.data, r[expected_blob.id].data)
    actual_commit = r[self._root_commit]
    self.assertEqual(b'msg', actual_commit.message)