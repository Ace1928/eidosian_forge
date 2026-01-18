import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_match_multiple_times3(self):
    tmpl = MarkupTemplate('<?xml version="1.0"?>\n          <root xmlns:py="http://genshi.edgewall.org/">\n            <py:match path="foo/bar">\n              <zzzzz/>\n            </py:match>\n            <foo>\n              <bar/>\n              <bar/>\n            </foo>\n            <bar/>\n          </root>')
    self.assertEqual('<?xml version="1.0"?>\n<root>\n            <foo>\n              <zzzzz/>\n              <zzzzz/>\n            </foo>\n            <bar/>\n          </root>', tmpl.generate().render())