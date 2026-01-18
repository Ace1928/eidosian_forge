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
def test_leading_newline(self):
    self.repl.send_to_stdouterr('\nfoo\n')
    self.assertEqual(self.repl.display_lines[-2], '')
    self.assertEqual(self.repl.display_lines[-1], 'foo')
    self.assertEqual(self.repl.current_stdouterr_line, '')