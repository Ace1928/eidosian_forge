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
def test_matcher_suppression_with_iterator(self):

    @completion_matcher(identifier='matcher_returning_iterator')
    def matcher_returning_iterator(text):
        return iter(['completion_iter'])

    @completion_matcher(identifier='matcher_returning_list')
    def matcher_returning_list(text):
        return ['completion_list']
    with custom_matchers([matcher_returning_iterator, matcher_returning_list]):
        ip = get_ipython()
        c = ip.Completer

        def _(text, expected):
            c.use_jedi = False
            s, matches = c.complete(text)
            self.assertEqual(expected, matches)

        def configure(suppression_config):
            cfg = Config()
            cfg.IPCompleter.suppress_competing_matchers = suppression_config
            c.update_config(cfg)
        configure(False)
        _('---', ['completion_iter', 'completion_list'])
        configure(True)
        _('---', ['completion_iter'])
        configure(None)
        _('--', ['completion_iter', 'completion_list'])