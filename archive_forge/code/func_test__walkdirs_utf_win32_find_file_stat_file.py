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
def test__walkdirs_utf_win32_find_file_stat_file(self):
    """make sure our Stat values are valid"""
    self.requireFeature(test__walkdirs_win32.win32_readdir_feature)
    self.requireFeature(features.UnicodeFilenameFeature)
    from .._walkdirs_win32 import Win32ReadDir
    name0u = '0file-¶'
    name0 = name0u.encode('utf8')
    self.build_tree([name0u])
    time.sleep(2)
    with open(name0u, 'ab') as f:
        f.write(b'just a small update')
    result = Win32ReadDir().read_dir('', '.')
    entry = result[0]
    self.assertEqual((name0, name0, 'file'), entry[:3])
    self.assertEqual('./' + name0u, entry[4])
    self.assertStatIsCorrect(entry[4], entry[3])
    self.assertNotEqual(entry[3].st_mtime, entry[3].st_ctime)