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
def test_init_existing(self):
    tmp_dir = self.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp_dir)
    t = Repo.init(tmp_dir)
    self.addCleanup(t.close)
    self.assertEqual(os.listdir(tmp_dir), ['.git'])
    self.assertFilesystemHidden(os.path.join(tmp_dir, '.git'))