import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def test_iexplore(self):
    for a in ('iexplore', 'iexplore.exe'):
        p = get_app_path(a)
        d, b = os.path.split(p)
        self.assertEqual('iexplore.exe', b.lower())
        self.assertNotEqual('', d)