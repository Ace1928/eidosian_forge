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
def test_line_split():
    """Basic line splitter test with default specs."""
    sp = completer.CompletionSplitter()
    t = [('run some/scrip', '', 'some/scrip'), ('run scripts/er', 'ror.py foo', 'scripts/er'), ('echo $HOM', '', 'HOM'), ('print sys.pa', '', 'sys.pa'), ('print(sys.pa', '', 'sys.pa'), ("execfile('scripts/er", '', 'scripts/er'), ('a[x.', '', 'x.'), ('a[x.', 'y', 'x.'), ('cd "some_file/', '', 'some_file/')]
    check_line_split(sp, t)
    check_line_split(sp, [map(str, p) for p in t])