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
def test_current(self):
    with self.assertRaises(ValueError):
        self.matches_iterator.current()
    next(self.matches_iterator)
    self.assertEqual(self.matches_iterator.current(), self.matches[0])