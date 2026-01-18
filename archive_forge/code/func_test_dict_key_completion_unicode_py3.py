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
def test_dict_key_completion_unicode_py3(self):
    """Test handling of unicode in dict key completion"""
    ip = get_ipython()
    complete = ip.Completer.complete
    ip.user_ns['d'] = {'aא': None}
    if sys.platform != 'win32':
        _, matches = complete(line_buffer="d['a\\u05d0")
        self.assertIn('u05d0', matches)
    _, matches = complete(line_buffer="d['aא")
    self.assertIn('aא', matches)
    with greedy_completion():
        _, matches = complete(line_buffer="d['a\\u05d0")
        self.assertIn("d['a\\u05d0']", matches)
        _, matches = complete(line_buffer="d['aא")
        self.assertIn("d['aא']", matches)