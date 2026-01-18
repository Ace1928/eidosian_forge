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
def test_lambda_position(self):
    self.set_input_line('spam(lambda a, b: 1, ')
    self.assertTrue(self.repl.get_args())
    self.assertTrue(self.repl.funcprops)
    self.assertEqual(self.repl.arg_pos, 1)