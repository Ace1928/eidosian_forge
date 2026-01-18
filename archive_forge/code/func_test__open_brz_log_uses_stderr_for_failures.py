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
def test__open_brz_log_uses_stderr_for_failures(self):
    self.overrideAttr(sys, 'stderr', StringIO())
    self.overrideEnv('BRZ_LOG', '/no-such-dir/brz.log')
    self.overrideAttr(trace, '_brz_log_filename')
    logf = trace._open_brz_log()
    if os.path.isdir('/no-such-dir'):
        raise TestSkipped('directory creation succeeded')
    self.assertIs(None, logf)
    self.assertContainsRe(sys.stderr.getvalue(), "failed to open trace file: .* '/no-such-dir/brz.log'$")