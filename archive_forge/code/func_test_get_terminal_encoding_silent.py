import codecs
import locale
import sys
from typing import Set
from .. import osutils
from . import TestCase
from .ui_testing import BytesIOWithEncoding, StringIOWithEncoding
def test_get_terminal_encoding_silent(self):
    self.make_wrapped_streams('stdout_encoding', 'stderr_encoding', 'stdin_encoding')
    log = self.get_log()
    osutils.get_terminal_encoding()
    self.assertEqual(log, self.get_log())