import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_cascaded_matches(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <body py:match="body">${select(\'*\')}</body>\n          <head py:match="head">${select(\'title\')}</head>\n          <body py:match="body">${select(\'*\')}<hr /></body>\n          <head><title>Welcome to Markup</title></head>\n          <body><h2>Are you ready to mark up?</h2></body>\n        </html>')
    self.assertEqual('<html>\n          <head><title>Welcome to Markup</title></head>\n          <body><h2>Are you ready to mark up?</h2><hr/></body>\n        </html>', tmpl.generate().render(encoding=None))