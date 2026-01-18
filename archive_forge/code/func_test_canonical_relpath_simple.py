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
def test_canonical_relpath_simple(self):
    f = open('MixedCaseName', 'w')
    f.close()
    actual = osutils.canonical_relpath(self.test_base_dir, 'mixedcasename')
    self.assertEqual('work/MixedCaseName', actual)