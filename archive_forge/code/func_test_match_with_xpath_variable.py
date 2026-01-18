import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_match_with_xpath_variable(self):
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <span py:match="*[name()=$tagname]">\n            Hello ${select(\'@name\')}\n          </span>\n          <greeting name="Dude"/>\n        </div>')
    self.assertEqual('<div>\n          <span>\n            Hello Dude\n          </span>\n        </div>', tmpl.generate(tagname='greeting').render(encoding=None))
    self.assertEqual('<div>\n          <greeting name="Dude"/>\n        </div>', tmpl.generate(tagname='sayhello').render(encoding=None))