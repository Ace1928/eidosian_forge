import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_match_without_closure(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <p py:match="body/p" class="para">${select(\'*|text()\')}</p>\n          <body>\n            <p>Foo</p>\n            <div><p>Bar</p></div>\n          </body>\n        </html>')
    self.assertEqual('<html>\n          <body>\n            <p class="para">Foo</p>\n            <div><p>Bar</p></div>\n          </body>\n        </html>', tmpl.generate().render(encoding=None))