import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def assertMatch(self, matchset, glob_prefix=None):
    for glob, positive, negative in matchset:
        if glob_prefix:
            glob = glob_prefix + glob
        globster = Globster([glob])
        for name in positive:
            self.assertTrue(globster.match(name), repr('name "%s" does not match glob "%s" (re=%s)' % (name, glob, globster._regex_patterns[0][0].pattern)))
        for name in negative:
            self.assertFalse(globster.match(name), repr('name "%s" does match glob "%s" (re=%s)' % (name, glob, globster._regex_patterns[0][0].pattern)))