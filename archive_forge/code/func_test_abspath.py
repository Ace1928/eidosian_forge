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
def test_abspath(self):
    self.requireFeature(features.win32_feature)
    self.assertEqual('C:/foo', osutils._win32_abspath('C:\\foo'))
    self.assertEqual('C:/foo', osutils._win32_abspath('C:/foo'))
    self.assertEqual('//HOST/path', osutils._win32_abspath('\\\\HOST\\path'))
    self.assertEqual('//HOST/path', osutils._win32_abspath('//HOST/path'))