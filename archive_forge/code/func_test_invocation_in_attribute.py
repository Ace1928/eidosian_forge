import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_invocation_in_attribute(self):
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:def function="echo(what)">${what or \'something\'}</py:def>\n          <p class="${echo(\'foo\')}">bar</p>\n        </doc>')
    self.assertEqual('<doc>\n          <p class="foo">bar</p>\n        </doc>', tmpl.generate().render(encoding=None))