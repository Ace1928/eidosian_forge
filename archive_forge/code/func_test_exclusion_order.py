import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_exclusion_order(self):
    """test that ordering of exclusion patterns does not matter"""
    patterns = ['static/**/*.html', '!static/**/versionable.html']
    globster = ExceptionGlobster(patterns)
    self.assertEqual('static/**/*.html', globster.match('static/foo.html'))
    self.assertEqual(None, globster.match('static/versionable.html'))
    self.assertEqual(None, globster.match('static/bar/versionable.html'))
    globster = ExceptionGlobster(reversed(patterns))
    self.assertEqual('static/**/*.html', globster.match('static/foo.html'))
    self.assertEqual(None, globster.match('static/versionable.html'))
    self.assertEqual(None, globster.match('static/bar/versionable.html'))