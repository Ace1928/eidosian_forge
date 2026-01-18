import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_multiple_vars_trailing_semicolon(self):
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <py:with vars="x = x * 2; y = x / 2;">${x} ${y}</py:with>\n        </div>')
    self.assertEqual('<div>\n          84 %s\n        </div>' % (84 / 2), tmpl.generate(x=42).render(encoding=None))