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
def test_fill_screen(self):
    self.repl.width, self.repl.height = (20, 15)
    self.locals['abc'] = completion_target(20, 100)
    self.repl.current_line = 'abc'
    self.repl.cursor_offset = 3
    self.repl.process_event('.')
    screen = self.process_box_characters(['>>> abc.', '┌──────────────────┐', '│ aaaaaaaaaaaaaaaa │', '│ b                │', '│ c                │', '│ d                │', '│ e                │', '│ f                │', '│ g                │', '│ h                │', '│ i                │', '│ j                │', '│ k                │', '│ l                │', '└──────────────────┘'])
    self.assert_paint_ignoring_formatting(screen, (0, 8))