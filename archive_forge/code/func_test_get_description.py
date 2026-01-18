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
def test_get_description(self):
    r = self.open_repo('a.git')
    with open(os.path.join(r.path, 'description'), 'wb') as f:
        f.write(b'Some description')
    self.assertEqual(b'Some description', r.get_description())