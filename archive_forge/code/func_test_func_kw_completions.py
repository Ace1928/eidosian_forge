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
def test_func_kw_completions(self):
    ip = get_ipython()
    c = ip.Completer
    c.use_jedi = False
    ip.ex('def myfunc(a=1,b=2): return a+b')
    s, matches = c.complete(None, 'myfunc(1,b')
    self.assertIn('b=', matches)
    s, matches = c.complete(None, 'myfunc(1,b)', 10)
    self.assertIn('b=', matches)
    s, matches = c.complete(None, 'myfunc(a="escaped\\")string",b')
    self.assertIn('b=', matches)
    s, matches = c.complete(None, 'min(k, k')
    self.assertIn('key=', matches)