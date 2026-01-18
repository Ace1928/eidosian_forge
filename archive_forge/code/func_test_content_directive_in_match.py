import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_content_directive_in_match(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <div py:match="foo">I said <q py:content="select(\'text()\')">something</q>.</div>\n          <foo>bar</foo>\n        </html>')
    self.assertEqual('<html>\n          <div>I said <q>bar</q>.</div>\n        </html>', tmpl.generate().render(encoding=None))