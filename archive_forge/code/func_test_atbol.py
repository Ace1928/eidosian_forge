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
def test_atbol(self):
    self.assertTrue(self.repl.atbol())
    self.repl.s = '\t\t'
    self.assertTrue(self.repl.atbol())
    self.repl.s = '\t\tnot an empty line'
    self.assertFalse(self.repl.atbol())