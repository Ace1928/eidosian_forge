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
def test_mutter_never_fails(self):
    """Even with unencodable input mutter should not raise errors."""
    mutter('can write unicode §')
    mutter('can interpolate unicode %s', '§')
    mutter(b'can write bytes \xa7')
    mutter('can repr bytes %r', b'\xa7')
    mutter('can interpolate bytes %s', b'\xa7')
    log = self.get_log()
    self.assertContainsRe(log, ".* +can write unicode §\n.* +can interpolate unicode §\n.* +can write bytes �\n.* +can repr bytes b'\\\\xa7'\n.* +can interpolate bytes (?:�|b'\\\\xa7')\n")