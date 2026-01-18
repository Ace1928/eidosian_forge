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
def set_input_line(self, line):
    """Set current input line of the test REPL."""
    self.repl.current_line = line
    self.repl.cursor_offset = len(line)