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
def test_completion(self):
    self.repl.height, self.repl.width = (5, 32)
    self.repl.current_line = 'an'
    self.cursor_offset = 2
    screen = self.process_box_characters(['>>> an', '┌──────────────────────────────┐', '│ and  any(                    │', '└──────────────────────────────┘', 'Welcome to bpython! Press <F1> f'] if sys.version_info[:2] < (3, 10) else ['>>> an', '┌──────────────────────────────┐', '│ and    anext( any(           │', '└──────────────────────────────┘', 'Welcome to bpython! Press <F1> f'])
    self.assert_paint_ignoring_formatting(screen, (0, 4))