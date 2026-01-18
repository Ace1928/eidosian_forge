import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_nested_vars_single_assignment(self):
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <py:with vars="x, (y, z) = (1, (2, 3))">${x} ${y} ${z}</py:with>\n        </div>')
    self.assertEqual('<div>\n          1 2 3\n        </div>', tmpl.generate(x=42).render(encoding=None))