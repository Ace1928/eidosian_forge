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
def test_format_memory_error(self):
    try:
        raise MemoryError()
    except MemoryError:
        msg = _format_exception()
    self.assertEqual(msg, 'brz: out of memory\nUse -Dmem_dump to dump memory to a file.\n')