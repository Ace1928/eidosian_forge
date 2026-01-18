import itertools
import os
import pydoc
import string
import sys
from contextlib import contextmanager
from typing import cast
from curtsies.formatstringarray import (
from curtsies.fmtfuncs import cyan, bold, green, yellow, on_magenta, red
from curtsies.window import CursorAwareWindow
from unittest import mock, skipIf
from bpython.curtsiesfrontend.events import RefreshRequestEvent
from bpython import config, inspection
from bpython.curtsiesfrontend.repl import BaseRepl
from bpython.curtsiesfrontend import replpainter
from bpython.curtsiesfrontend.repl import (
from bpython.test import FixLanguageTestCase as TestCase, TEST_CONFIG
@skipIf(sys.version_info[:2] >= (3, 11) and sys.version_info[:3] < (3, 11, 1), 'https://github.com/python/cpython/issues/98744')
def test_display_of_padding_chars(self):
    self.repl.width = 11
    [self.repl.add_normal_character(c) for c in 'ｗｉｄｔｈ']
    self.enter()
    expected = ['>>> ｗｉｄ ', 'ｔｈ']
    result = [d.s for d in self.repl.display_lines[0:2]]
    self.assertEqual(result, expected)