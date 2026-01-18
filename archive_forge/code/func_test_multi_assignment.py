import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_multi_assignment(self):
    """
        Verify that assignment to tuples works correctly.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:for each="k, v in items">\n            <p>key=$k, value=$v</p>\n          </py:for>\n        </doc>')
    self.assertEqual('<doc>\n            <p>key=a, value=1</p>\n            <p>key=b, value=2</p>\n        </doc>', tmpl.generate(items=(('a', 1), ('b', 2))).render(encoding=None))