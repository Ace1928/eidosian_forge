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
def test_get_parents(self):
    r = self.open_repo('a.git')
    self.assertEqual([b'2a72d929692c41d8554c07f6301757ba18a65d91'], r.get_parents(b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'))
    r.update_shallow([b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'], None)
    self.assertEqual([], r.get_parents(b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'))