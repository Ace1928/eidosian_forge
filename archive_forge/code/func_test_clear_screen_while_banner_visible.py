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
def test_clear_screen_while_banner_visible(self):
    self.repl.status_bar.message('STATUS_BAR')
    self.enter('1 + 1')
    self.enter('2 + 2')
    screen = ['>>> 1 + 1', '2', '>>> 2 + 2', '4', '>>> ', 'STATUS_BAR                      ']
    self.assert_paint_ignoring_formatting(screen, (4, 4))
    self.repl.scroll_offset += len(screen) - self.repl.height
    self.assert_paint_ignoring_formatting(screen[1:], (3, 4))
    self.repl.request_paint_to_clear_screen = True
    screen = ['2', '>>> 2 + 2', '4', '>>> ', '', '', '', 'STATUS_BAR                      ']
    self.assert_paint_ignoring_formatting(screen, (3, 4))