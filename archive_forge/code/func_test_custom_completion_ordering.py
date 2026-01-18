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
def test_custom_completion_ordering(self):
    """Test that errors from custom attribute completers are silenced."""
    ip = get_ipython()
    _, matches = ip.complete('in')
    assert matches.index('input') < matches.index('int')

    def complete_example(a):
        return ['example2', 'example1']
    ip.Completer.custom_completers.add_re('ex*', complete_example)
    _, matches = ip.complete('ex')
    assert matches.index('example2') < matches.index('example1')