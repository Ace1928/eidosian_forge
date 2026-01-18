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
def dir_reader_scenarios():
    scenarios = [('unicode', dict(_dir_reader_class=osutils.UnicodeDirReader, _native_to_unicode=_already_unicode))]
    if UTF8DirReaderFeature.available():
        from .. import _readdir_pyx
        scenarios.append(('utf8', dict(_dir_reader_class=_readdir_pyx.UTF8DirReader, _native_to_unicode=_utf8_to_unicode)))
    if test__walkdirs_win32.win32_readdir_feature.available():
        try:
            from .. import _walkdirs_win32
            scenarios.append(('win32', dict(_dir_reader_class=_walkdirs_win32.Win32ReadDir, _native_to_unicode=_already_unicode)))
        except ModuleNotFoundError:
            pass
    return scenarios