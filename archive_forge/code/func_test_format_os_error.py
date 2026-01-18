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
def test_format_os_error(self):
    try:
        os.rmdir('nosuchfile22222')
    except OSError as e:
        e_str = str(e)
        msg = _format_exception()
    self.assertEqual('brz: ERROR: {}\n'.format(e_str), msg)