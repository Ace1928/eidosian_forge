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
def test_get_shallow(self):
    self.assertEqual(set(), self._repo.get_shallow())
    with open(os.path.join(self._repo.path, '.git', 'shallow'), 'wb') as f:
        f.write(b'a90fa2d900a17e99b433217e988c4eb4a2e9a097\n')
    self.assertEqual({b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'}, self._repo.get_shallow())