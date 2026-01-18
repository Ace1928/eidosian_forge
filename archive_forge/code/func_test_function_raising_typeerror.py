import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_function_raising_typeerror(self):

    def badfunc():
        raise TypeError
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <div py:def="dobadfunc()">\n            ${badfunc()}\n          </div>\n          <div py:content="dobadfunc()"/>\n        </html>')
    self.assertRaises(TypeError, list, tmpl.generate(badfunc=badfunc))