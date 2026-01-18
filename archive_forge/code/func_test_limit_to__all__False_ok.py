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
def test_limit_to__all__False_ok(self):
    """
        Limit to all is deprecated, once we remove it this test can go away. 
        """
    ip = get_ipython()
    c = ip.Completer
    c.use_jedi = False
    ip.ex('class D: x=24')
    ip.ex('d=D()')
    cfg = Config()
    cfg.IPCompleter.limit_to__all__ = False
    c.update_config(cfg)
    s, matches = c.complete('d.')
    self.assertIn('d.x', matches)