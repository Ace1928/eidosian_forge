import codecs
import locale
import sys
from typing import Set
from .. import osutils
from . import TestCase
from .ui_testing import BytesIOWithEncoding, StringIOWithEncoding
def test_get_terminal_encoding(self):
    self.make_wrapped_streams('stdout_encoding', 'stderr_encoding', 'stdin_encoding')
    self.assertEqual('stdout_encoding', osutils.get_terminal_encoding())
    sys.stdout.encoding = None
    self.assertEqual('stdin_encoding', osutils.get_terminal_encoding())
    sys.stdin.encoding = None
    self.assertEqual('user_encoding', osutils.get_terminal_encoding())