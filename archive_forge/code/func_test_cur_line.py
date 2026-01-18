import collections
import inspect
import socket
import sys
import tempfile
import unittest
from typing import List, Tuple
from itertools import islice
from pathlib import Path
from unittest import mock
from bpython import config, repl, cli, autocomplete
from bpython.line import LinePart
from bpython.test import (
def test_cur_line(self):
    completer = mock.Mock()
    completer.locate.return_value = LinePart(0, self.matches_iterator.orig_cursor_offset, self.matches_iterator.orig_line)
    self.matches_iterator.completer = completer
    with self.assertRaises(ValueError):
        self.matches_iterator.cur_line()
    self.assertEqual(next(self.matches_iterator), self.matches[0])
    self.assertEqual(self.matches_iterator.cur_line(), (len(self.matches[0]), self.matches[0]))