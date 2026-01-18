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
def test_line_magics(self):
    ip = get_ipython()
    c = ip.Completer
    s, matches = c.complete(None, 'lsmag')
    self.assertIn('%lsmagic', matches)
    s, matches = c.complete(None, '%lsmag')
    self.assertIn('%lsmagic', matches)