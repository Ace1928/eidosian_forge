import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def assertCommandLine(self, expected, line, argv=None, single_quotes_allowed=False):
    if argv is None:
        argv = [line]
    argv = win32utils._command_line_to_argv(line, argv, single_quotes_allowed=single_quotes_allowed)
    self.assertEqual(expected, sorted(argv))