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
def test_default_arguments_from_docstring(self):
    ip = get_ipython()
    c = ip.Completer
    kwd = c._default_arguments_from_docstring('min(iterable[, key=func]) -> value')
    self.assertEqual(kwd, ['key'])
    kwd = c._default_arguments_from_docstring('Minuit.migrad(self, int ncall=10000, resume=True, int nsplit=1)\n')
    self.assertEqual(kwd, ['ncall', 'resume', 'nsplit'])
    kwd = c._default_arguments_from_docstring('\n Minuit.migrad(self, int ncall=10000, resume=True, int nsplit=1)\n')
    self.assertEqual(kwd, ['ncall', 'resume', 'nsplit'])