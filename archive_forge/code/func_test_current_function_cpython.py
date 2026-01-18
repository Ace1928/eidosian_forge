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
@unittest.skipIf(pypy, 'different errors for PyPy')
def test_current_function_cpython(self):
    self.set_input_line('INPUTLINE')
    self.assert_get_source_error_for_current_function(collections.defaultdict.copy, 'No source code found for INPUTLINE')
    self.assert_get_source_error_for_current_function(collections.defaultdict, 'could not find class definition')