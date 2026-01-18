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
def test_enter_text(self):
    [self.repl.add_normal_character(c) for c in '1 + 1']
    screen = fsarray([cyan('>>> ') + bold(green('1') + cyan(' ') + yellow('+') + cyan(' ') + green('1')), cyan('Welcome to')])
    self.assert_paint(screen, (0, 9))