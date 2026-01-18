import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_multiple_vars_single_assignment(self):
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <py:with vars="x = y = z = 1">${x} ${y} ${z}</py:with>\n        </div>')
    self.assertEqual('<div>\n          1 1 1\n        </div>', tmpl.generate(x=42).render(encoding=None))