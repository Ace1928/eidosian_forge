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
def test_force_walkdirs_utf8_fs_ascii(self):
    self.requireFeature(UTF8DirReaderFeature)
    self._save_platform_info()
    self.assertDirReaderIs(UTF8DirReaderFeature.module.UTF8DirReader, b'.', fs_enc='ascii')