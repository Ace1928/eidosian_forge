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
def test_latex_no_results(self):
    """
        forward latex should really return nothing in either field if nothing is found.
        """
    ip = get_ipython()
    text, matches = ip.Completer.latex_matches('\\really_i_should_match_nothing')
    self.assertEqual(text, '')
    self.assertEqual(matches, ())