import code
import os
import sys
import tempfile
import io
from typing import cast
import unittest
from contextlib import contextmanager
from functools import partial
from unittest import mock
from bpython.curtsiesfrontend import repl as curtsiesrepl
from bpython.curtsiesfrontend import interpreter
from bpython.curtsiesfrontend import events as bpythonevents
from bpython.curtsiesfrontend.repl import LineType
from bpython import autocomplete
from bpython import config
from bpython import args
from bpython.test import (
from curtsies import events
from curtsies.window import CursorAwareWindow
from importlib import invalidate_caches
@unittest.skipUnless(all(map(config.can_encode, 'å∂ßƒ')), 'Charset can not encode characters')
def test_external_communication_encoding(self):
    with captured_output():
        self.repl.display_lines.append('>>> "åß∂ƒ"')
        self.repl.history.append('"åß∂ƒ"')
        self.repl.all_logical_lines.append(('"åß∂ƒ"', LineType.INPUT))
        self.repl.send_session_to_external_editor()