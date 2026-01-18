import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def build_ascii_tree(self):
    self.build_tree(['a', 'a1', 'a2', 'a11', 'a.1', 'b', 'b1', 'b2', 'b3', 'c/', 'c/c1', 'c/c2', 'd/', 'd/d1', 'd/d2', 'd/e/', 'd/e/e1'])