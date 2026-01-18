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
def test_kind_marker(self):
    self.assertEqual('', osutils.kind_marker('file'))
    self.assertEqual('/', osutils.kind_marker('directory'))
    self.assertEqual('/', osutils.kind_marker(osutils._directory_kind))
    self.assertEqual('@', osutils.kind_marker('symlink'))
    self.assertEqual('+', osutils.kind_marker('tree-reference'))
    self.assertEqual('', osutils.kind_marker('fifo'))
    self.assertEqual('', osutils.kind_marker('socket'))
    self.assertEqual('', osutils.kind_marker('unknown'))