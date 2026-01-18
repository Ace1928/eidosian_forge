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
def test_cursor_stays_at_bottom_of_screen(self):
    """infobox showing up during intermediate render was causing this to
        fail, #371"""
    self.repl.width = 50
    self.repl.current_line = "__import__('random').__name__"
    with output_to_repl(self.repl):
        self.repl.on_enter(new_code=False)
    screen = [">>> __import__('random').__name__", "'random'"]
    self.assert_paint_ignoring_formatting(screen)
    with output_to_repl(self.repl):
        self.repl.process_event(self.refresh_requests.pop())
    screen = [">>> __import__('random').__name__", "'random'", '']
    self.assert_paint_ignoring_formatting(screen)
    with output_to_repl(self.repl):
        self.repl.process_event(self.refresh_requests.pop())
    screen = [">>> __import__('random').__name__", "'random'", '>>> ']
    self.assert_paint_ignoring_formatting(screen, (2, 4))