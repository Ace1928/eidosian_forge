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
def test_matcher_suppression(self):

    @completion_matcher(identifier='a_matcher')
    def a_matcher(text):
        return ['completion_a']

    @completion_matcher(identifier='b_matcher', api_version=2)
    def b_matcher(context: CompletionContext):
        text = context.token
        result = {'completions': [SimpleCompletion('completion_b')]}
        if text == 'suppress c':
            result['suppress'] = {'c_matcher'}
        if text.startswith('suppress all'):
            result['suppress'] = True
            if text == 'suppress all but c':
                result['do_not_suppress'] = {'c_matcher'}
            if text == 'suppress all but a':
                result['do_not_suppress'] = {'a_matcher'}
        return result

    @completion_matcher(identifier='c_matcher')
    def c_matcher(text):
        return ['completion_c']
    with custom_matchers([a_matcher, b_matcher, c_matcher]):
        ip = get_ipython()
        c = ip.Completer

        def _(text, expected):
            c.use_jedi = False
            s, matches = c.complete(text)
            self.assertEqual(expected, matches)
        _('do not suppress', ['completion_a', 'completion_b', 'completion_c'])
        _('suppress all', ['completion_b'])
        _('suppress all but a', ['completion_a', 'completion_b'])
        _('suppress all but c', ['completion_b', 'completion_c'])

        def configure(suppression_config):
            cfg = Config()
            cfg.IPCompleter.suppress_competing_matchers = suppression_config
            c.update_config(cfg)
        configure(False)
        _('suppress all', ['completion_a', 'completion_b', 'completion_c'])
        configure({'b_matcher': False})
        _('suppress all', ['completion_a', 'completion_b', 'completion_c'])
        configure({'a_matcher': False})
        _('suppress all', ['completion_b'])
        configure({'b_matcher': True})
        _('do not suppress', ['completion_b'])
        configure(True)
        _('do not suppress', ['completion_a'])