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
def test_dict_key_completion_closures(self):
    ip = get_ipython()
    complete = ip.Completer.complete
    ip.Completer.auto_close_dict_keys = True
    ip.user_ns['d'] = {('aa', 11): None, ('bb', 22): None, 'bb': None, 'cc': None, (77, 'x'): None, (88, 'y'): None, 88: None, 99: None}
    _, matches = complete(line_buffer='d[')
    self.assertIn("'aa', ", matches)
    self.assertIn("'bb'", matches)
    self.assertIn("'cc']", matches)
    self.assertIn('77, ', matches)
    self.assertIn('88', matches)
    self.assertIn('99]', matches)
    _, matches = complete(line_buffer="d['aa', ")
    self.assertIn('11]', matches)
    self.assertNotIn("'bb'", matches)
    self.assertNotIn("'bb', ", matches)
    self.assertNotIn("'bb']", matches)
    self.assertNotIn("'cc'", matches)
    self.assertNotIn("'cc', ", matches)
    self.assertNotIn("'cc']", matches)
    ip.Completer.auto_close_dict_keys = False