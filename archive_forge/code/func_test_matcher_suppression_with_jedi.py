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
@pytest.mark.xfail(sys.version_info.releaselevel in ('alpha',), reason='Parso does not yet parse 3.13')
def test_matcher_suppression_with_jedi(self):
    ip = get_ipython()
    c = ip.Completer
    c.use_jedi = True

    def configure(suppression_config):
        cfg = Config()
        cfg.IPCompleter.suppress_competing_matchers = suppression_config
        c.update_config(cfg)

    def _():
        with provisionalcompleter():
            matches = [completion.text for completion in c.completions('dict.', 5)]
            self.assertIn('keys', matches)
    configure(False)
    _()
    configure(True)
    _()
    configure(None)
    _()