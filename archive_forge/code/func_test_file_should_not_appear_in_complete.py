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
def test_file_should_not_appear_in_complete(self):
    self.repl = FakeRepl({'autocomplete_mode': autocomplete.AutocompleteModes.SIMPLE})
    self.set_input_line('_')
    self.assertTrue(self.repl.complete())
    self.assertTrue(hasattr(self.repl.matches_iter, 'matches'))
    self.assertNotIn('__file__', self.repl.matches_iter.matches)