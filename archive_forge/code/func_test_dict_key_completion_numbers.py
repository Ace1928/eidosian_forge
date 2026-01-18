import os
import pytest
import sys
import textwrap
import unittest
from contextlib import contextmanager
from traitlets.config.loader import Config
from IPython import get_ipython
from IPython.core import completer
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.generics import complete_object
from IPython.testing import decorators as dec
from IPython.core.completer import (
def test_dict_key_completion_numbers(self):
    ip = get_ipython()
    complete = ip.Completer.complete
    ip.user_ns['d'] = {3735928559: None, 1111: None, 1234: None, '1999': None, 21: None, 22: None}
    _, matches = complete(line_buffer='d[1')
    self.assertIn('1111', matches)
    self.assertIn('1234', matches)
    self.assertNotIn('1999', matches)
    self.assertNotIn("'1999'", matches)
    _, matches = complete(line_buffer='d[0xdead')
    self.assertIn('0xdeadbeef', matches)
    _, matches = complete(line_buffer='d[2')
    self.assertIn('21', matches)
    self.assertIn('22', matches)
    _, matches = complete(line_buffer='d[0b101')
    self.assertIn('0b10101', matches)
    self.assertIn('0b10110', matches)