import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_def_in_matched(self):
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <head py:match="head">${select(\'*\')}</head>\n          <head>\n            <py:def function="maketitle(test)"><b py:replace="test" /></py:def>\n            <title>${maketitle(True)}</title>\n          </head>\n        </doc>')
    self.assertEqual('<doc>\n          <head><title>True</title></head>\n        </doc>', tmpl.generate().render(encoding=None))