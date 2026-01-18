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
def completes_on_nested():
    ip.user_ns['d'] = numpy.zeros(2, dtype=dt)
    _, matches = complete(line_buffer="d[1]['my_head']['")
    self.assertTrue(any(['my_dt' in m for m in matches]))
    self.assertTrue(any(['my_df' in m for m in matches]))