import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
def assertPathCompare(self, path_less, path_greater):
    """check that path_less and path_greater compare correctly."""
    self.assertEqual(0, osutils.compare_paths_prefix_order(path_less, path_less))
    self.assertEqual(0, osutils.compare_paths_prefix_order(path_greater, path_greater))
    self.assertEqual(-1, osutils.compare_paths_prefix_order(path_less, path_greater))
    self.assertEqual(1, osutils.compare_paths_prefix_order(path_greater, path_less))