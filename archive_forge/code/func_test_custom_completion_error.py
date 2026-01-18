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
def test_custom_completion_error(self):
    """Test that errors from custom attribute completers are silenced."""
    ip = get_ipython()

    class A:
        pass
    ip.user_ns['x'] = A()

    @complete_object.register(A)
    def complete_A(a, existing_completions):
        raise TypeError('this should be silenced')
    ip.complete('x.')