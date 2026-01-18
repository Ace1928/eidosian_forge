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
@unittest.skipIf(pypy, "pypy pydoc doesn't have this")
def test_issue583(self):
    self.repl = FakeRepl()
    self.repl.push('a = 1.2\n', False)
    self.set_input_line('a.is_integer(')
    self.repl.set_docstring()
    self.assertIsNot(self.repl.docstring, None)