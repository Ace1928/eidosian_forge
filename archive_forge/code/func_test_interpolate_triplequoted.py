import doctest
import sys
import unittest
from genshi.core import TEXT
from genshi.template.base import TemplateSyntaxError, EXPR
from genshi.template.interpolation import interpolate
def test_interpolate_triplequoted(self):
    parts = list(interpolate('${"""foo\nbar"""}'))
    self.assertEqual(1, len(parts))
    self.assertEqual('"""foo\nbar"""', parts[0][1].source)