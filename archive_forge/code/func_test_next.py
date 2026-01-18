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
def test_next(self):
    self.assertEqual(next(self.matches_iterator), self.matches[0])
    for x in range(len(self.matches) - 1):
        next(self.matches_iterator)
    self.assertEqual(next(self.matches_iterator), self.matches[0])
    self.assertEqual(next(self.matches_iterator), self.matches[1])
    self.assertNotEqual(next(self.matches_iterator), self.matches[1])