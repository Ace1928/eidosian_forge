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
def test_nested_import_module_completer(self):
    ip = get_ipython()
    _, matches = ip.complete(None, 'import IPython.co', 17)
    self.assertIn('IPython.core', matches)
    self.assertNotIn('import IPython.core', matches)
    self.assertNotIn('IPython.display', matches)