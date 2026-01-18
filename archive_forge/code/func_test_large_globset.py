import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_large_globset(self):
    """tests that the globster can handle a large set of patterns.

        Large is defined as more than supported by python regex groups,
        i.e. 99.
        This test assumes the globs are broken into regexs containing 99
        groups.
        """
    patterns = ['*.%03d' % i for i in range(300)]
    globster = Globster(patterns)
    for x in (0, 98, 99, 197, 198, 296, 297, 299):
        filename = 'foo.%03d' % x
        self.assertEqual(patterns[x], globster.match(filename))
    self.assertEqual(None, globster.match('foobar.300'))