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
def test_file_kind(self):
    self.build_tree(['file', 'dir/'])
    self.assertEqual('file', osutils.file_kind('file'))
    self.assertEqual('directory', osutils.file_kind('dir/'))
    if osutils.supports_symlinks(self.test_dir):
        os.symlink('symlink', 'symlink')
        self.assertEqual('symlink', osutils.file_kind('symlink'))
    try:
        os.lstat('/dev/null')
    except OSError as e:
        if e.errno not in (errno.ENOENT,):
            raise
    else:
        self.assertEqual('chardev', osutils.file_kind(os.path.realpath('/dev/null')))
    mkfifo = getattr(os, 'mkfifo', None)
    if mkfifo:
        mkfifo('fifo')
        try:
            self.assertEqual('fifo', osutils.file_kind('fifo'))
        finally:
            os.remove('fifo')
    AF_UNIX = getattr(socket, 'AF_UNIX', None)
    if AF_UNIX:
        s = socket.socket(AF_UNIX)
        s.bind('socket')
        try:
            self.assertEqual('socket', osutils.file_kind('socket'))
        finally:
            os.remove('socket')