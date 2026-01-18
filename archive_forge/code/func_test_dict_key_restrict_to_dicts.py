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
def test_dict_key_restrict_to_dicts(self):
    """Test that dict key suppresses non-dict completion items"""
    ip = get_ipython()
    c = ip.Completer
    d = {'abc': None}
    ip.user_ns['d'] = d
    text = 'd["a'

    def _():
        with provisionalcompleter():
            c.use_jedi = True
            return [completion.text for completion in c.completions(text, len(text))]
    completions = _()
    self.assertEqual(completions, ['abc'])
    cfg = Config()
    cfg.IPCompleter.suppress_competing_matchers = {'IPCompleter.dict_key_matcher': False}
    c.update_config(cfg)
    completions = _()
    self.assertIn('abc', completions)
    self.assertGreater(len(completions), 1)