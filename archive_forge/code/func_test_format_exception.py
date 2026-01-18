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
def test_format_exception(self):
    """Short formatting of brz exceptions"""
    try:
        raise errors.NotBranchError('wibble')
    except errors.NotBranchError:
        msg = _format_exception()
    self.assertEqual(msg, 'brz: ERROR: Not a branch: "wibble".\n')