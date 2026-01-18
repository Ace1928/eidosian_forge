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
def test_log_utf8_msg(self):
    logging.getLogger('brz').debug(b'\xc2\xa7')
    self.assertEqual('   DEBUG  §\n', self.get_log())