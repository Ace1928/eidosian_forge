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
def test_log_bytes_arg(self):
    logging.getLogger('brz').debug(b'%s', b'\xa7')
    log = self.get_log()
    self.assertEqual("   DEBUG  b'\\xa7'\n", self.get_log())