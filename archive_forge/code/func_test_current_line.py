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
def test_current_line(self):
    self.repl.interp.locals['a'] = socket.socket
    self.set_input_line('a')
    self.assertIn('dup(self)', self.repl.get_source_of_current_name())