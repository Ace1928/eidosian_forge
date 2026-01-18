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
def test_clone_no_head(self):
    temp_dir = self.mkdtemp()
    self.addCleanup(shutil.rmtree, temp_dir)
    repo_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'testdata', 'repos')
    dest_dir = os.path.join(temp_dir, 'a.git')
    shutil.copytree(os.path.join(repo_dir, 'a.git'), dest_dir, symlinks=True)
    r = Repo(dest_dir)
    self.addCleanup(r.close)
    del r.refs[b'refs/heads/master']
    del r.refs[b'HEAD']
    t = r.clone(os.path.join(temp_dir, 'b.git'), mkdir=True)
    self.addCleanup(t.close)
    self.assertEqual({b'refs/tags/mytag': b'28237f4dc30d0d462658d6b937b08a0f0b6ef55a', b'refs/tags/mytag-packed': b'b0931cadc54336e78a1d980420e3268903b57a50'}, t.refs.as_dict())