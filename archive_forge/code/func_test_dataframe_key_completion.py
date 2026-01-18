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
@dec.skip_without('pandas')
def test_dataframe_key_completion(self):
    """Test dict key completion applies to pandas DataFrames"""
    import pandas
    ip = get_ipython()
    complete = ip.Completer.complete
    ip.user_ns['d'] = pandas.DataFrame({'hello': [1], 'world': [2]})
    _, matches = complete(line_buffer="d['")
    self.assertIn('hello', matches)
    self.assertIn('world', matches)
    _, matches = complete(line_buffer="d.loc[:, '")
    self.assertIn('hello', matches)
    self.assertIn('world', matches)
    _, matches = complete(line_buffer="d.loc[1:, '")
    self.assertIn('hello', matches)
    _, matches = complete(line_buffer="d.loc[1:1, '")
    self.assertIn('hello', matches)
    _, matches = complete(line_buffer="d.loc[1:1:-1, '")
    self.assertIn('hello', matches)
    _, matches = complete(line_buffer="d.loc[::, '")
    self.assertIn('hello', matches)