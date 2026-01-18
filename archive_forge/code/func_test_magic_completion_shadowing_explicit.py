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
def test_magic_completion_shadowing_explicit(self):
    """
        If the user try to complete a shadowed magic, and explicit % start should
        still return the completions.
        """
    ip = get_ipython()
    c = ip.Completer
    text, matches = c.complete('%mat')
    self.assertEqual(matches, ['%matplotlib'])
    ip.run_cell('matplotlib = 1')
    text, matches = c.complete('%mat')
    self.assertEqual(matches, ['%matplotlib'])