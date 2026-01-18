import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_function_with_star_args(self):
    """
        Verify that a named template function using "star arguments" works as
        expected.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:def="f(*args, **kwargs)">\n            ${repr(args)}\n            ${repr(sorted(kwargs.items()))}\n          </div>\n          ${f(1, 2, a=3, b=4)}\n        </doc>')
    self.assertEqual("<doc>\n          <div>\n            [1, 2]\n            [('a', 3), ('b', 4)]\n          </div>\n        </doc>", tmpl.generate().render(encoding=None))