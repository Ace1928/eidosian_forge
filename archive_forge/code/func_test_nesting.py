import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_nesting(self):
    """
        Verify that `py:choose` blocks can be nested:
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:choose="1">\n            <div py:when="1" py:choose="3">\n              <span py:when="2">2</span>\n              <span py:when="3">3</span>\n            </div>\n          </div>\n        </doc>')
    self.assertEqual('<doc>\n          <div>\n            <div>\n              <span>3</span>\n            </div>\n          </div>\n        </doc>', tmpl.generate().render(encoding=None))