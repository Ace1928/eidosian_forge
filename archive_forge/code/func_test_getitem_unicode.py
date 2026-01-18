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
def test_getitem_unicode(self):
    r = self.open_repo('a.git')
    test_keys = [(b'refs/heads/master', True), (b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', True), (b'11' * 19 + b'--', False)]
    for k, contained in test_keys:
        self.assertEqual(k in r, contained)
    if getattr(self, 'assertRaisesRegex', None):
        assertRaisesRegexp = self.assertRaisesRegex
    else:
        assertRaisesRegexp = self.assertRaisesRegexp
    for k, _ in test_keys:
        assertRaisesRegexp(TypeError, "'name' must be bytestring, not int", r.__getitem__, 12)