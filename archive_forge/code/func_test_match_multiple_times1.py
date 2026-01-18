import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_match_multiple_times1(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="body[@id=\'content\']/h2" />\n          <head py:match="head" />\n          <head py:match="head" />\n          <head />\n          <body />\n        </html>')
    self.assertEqual('<html>\n          <head/>\n          <body/>\n        </html>', tmpl.generate().render())