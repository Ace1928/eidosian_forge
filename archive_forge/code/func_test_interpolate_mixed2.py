import doctest
import sys
import unittest
from genshi.core import TEXT
from genshi.template.base import TemplateSyntaxError, EXPR
from genshi.template.interpolation import interpolate
def test_interpolate_mixed2(self):
    parts = list(interpolate('foo $bar baz'))
    self.assertEqual(3, len(parts))
    self.assertEqual(TEXT, parts[0][0])
    self.assertEqual('foo ', parts[0][1])
    self.assertEqual(EXPR, parts[1][0])
    self.assertEqual('bar', parts[1][1].source)
    self.assertEqual(TEXT, parts[2][0])
    self.assertEqual(' baz', parts[2][1])