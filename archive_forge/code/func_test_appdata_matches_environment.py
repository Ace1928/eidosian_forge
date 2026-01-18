import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def test_appdata_matches_environment(self):
    encoding = osutils.get_user_encoding()
    env_val = os.environ.get('APPDATA', None)
    if not env_val:
        raise TestSkipped('No APPDATA environment variable exists')
    self.assertPathsEqual(win32utils.get_appdata_location(), env_val.decode(encoding))