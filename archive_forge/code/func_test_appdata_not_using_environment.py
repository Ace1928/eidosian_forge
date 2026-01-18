import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def test_appdata_not_using_environment(self):
    first = win32utils.get_appdata_location()
    self.overrideEnv('APPDATA', None)
    self.assertPathsEqual(first, win32utils.get_appdata_location())