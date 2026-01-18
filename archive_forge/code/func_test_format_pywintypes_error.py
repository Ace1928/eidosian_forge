import errno
import logging
import os
import re
import sys
import tempfile
from io import StringIO
from .. import debug, errors, trace
from ..trace import (_rollover_trace_maybe, be_quiet, get_verbosity_level,
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_format_pywintypes_error(self):
    self.requireFeature(features.pywintypes)
    import pywintypes
    import win32file
    try:
        win32file.RemoveDirectory('nosuchfile22222')
    except pywintypes.error:
        msg = _format_exception()
    self.assertContainsRe(msg, "^brz: ERROR: \\(2, 'RemoveDirectory[AW]?', .*\\)")