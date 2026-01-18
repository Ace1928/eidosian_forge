import doctest
import sys
import unittest
from genshi.core import TEXT
from genshi.template.base import TemplateSyntaxError, EXPR
from genshi.template.interpolation import interpolate
def test_interpolate_short_containing_underscore(self):
    parts = list(interpolate('$foo_bar'))
    self.assertEqual(1, len(parts))
    self.assertEqual(EXPR, parts[0][0])
    self.assertEqual('foo_bar', parts[0][1].source)