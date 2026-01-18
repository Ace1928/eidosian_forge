import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_match_multiple_times2(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="body/div[@id=\'properties\']" />\n          <head py:match="head" />\n          <head py:match="head" />\n          <head/>\n          <body>\n            <div id="properties">Foo</div>\n          </body>\n        </html>')
    self.assertEqual('<html>\n          <head/>\n          <body>\n          </body>\n        </html>', tmpl.generate().render())