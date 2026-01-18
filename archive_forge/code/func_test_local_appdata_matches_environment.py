import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def test_local_appdata_matches_environment(self):
    lad = win32utils.get_local_appdata_location()
    env = os.environ.get('LOCALAPPDATA')
    if env:
        encoding = osutils.get_user_encoding()
        self.assertPathsEqual(lad, env.decode(encoding))