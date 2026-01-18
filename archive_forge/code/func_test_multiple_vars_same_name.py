import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_multiple_vars_same_name(self):
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <py:with vars="\n            foo = \'bar\';\n            foo = foo.replace(\'r\', \'z\')\n          ">\n            $foo\n          </py:with>\n        </div>')
    self.assertEqual('<div>\n            baz\n        </div>', tmpl.generate(x=42).render(encoding=None))