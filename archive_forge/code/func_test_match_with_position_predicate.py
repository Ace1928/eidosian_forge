import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_match_with_position_predicate(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <p py:match="body/p[1]" class="first">${select(\'*|text()\')}</p>\n          <body>\n            <p>Foo</p>\n            <p>Bar</p>\n          </body>\n        </html>')
    self.assertEqual('<html>\n          <body>\n            <p class="first">Foo</p>\n            <p>Bar</p>\n          </body>\n        </html>', tmpl.generate().render(encoding=None))