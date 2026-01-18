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
def test_forward_unicode_completion(self):
    ip = get_ipython()
    name, matches = ip.complete('\\ROMAN NUMERAL FIVE')
    self.assertEqual(matches, ['Ⅴ'])
    self.assertEqual(matches, ['Ⅴ'])