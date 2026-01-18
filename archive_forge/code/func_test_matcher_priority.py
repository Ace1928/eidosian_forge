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
def test_matcher_priority(self):

    @completion_matcher(identifier='a_matcher', priority=0, api_version=2)
    def a_matcher(text):
        return {'completions': [SimpleCompletion('completion_a')], 'suppress': True}

    @completion_matcher(identifier='b_matcher', priority=2, api_version=2)
    def b_matcher(text):
        return {'completions': [SimpleCompletion('completion_b')], 'suppress': True}

    def _(expected):
        s, matches = c.complete('completion_')
        self.assertEqual(expected, matches)
    with custom_matchers([a_matcher, b_matcher]):
        ip = get_ipython()
        c = ip.Completer
        _(['completion_b'])
        a_matcher.matcher_priority = 3
        _(['completion_a'])