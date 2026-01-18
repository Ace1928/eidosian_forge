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
@unittest.skipIf(pypy, 'range pydoc has no signature in pypy')
def test_issue127(self):
    self.set_input_line('x=range(')
    self.assertTrue(self.repl.get_args())
    self.assertEqual(self.repl.current_func.__name__, 'range')
    self.set_input_line('{x:range(')
    self.assertTrue(self.repl.get_args())
    self.assertEqual(self.repl.current_func.__name__, 'range')
    self.set_input_line('foo(1, 2, x,range(')
    self.assertEqual(self.repl.current_func.__name__, 'range')
    self.set_input_line('(x,range(')
    self.assertEqual(self.repl.current_func.__name__, 'range')