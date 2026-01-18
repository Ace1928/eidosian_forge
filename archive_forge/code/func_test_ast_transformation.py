import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_ast_transformation(self):
    """
        Verify that the usual template expression AST transformations are
        applied despite the code being compiled to a `Suite` object.
        """
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <span py:with="bar=foo.bar">\n            $bar\n          </span>\n        </div>')
    self.assertEqual('<div>\n          <span>\n            42\n          </span>\n        </div>', tmpl.generate(foo={'bar': 42}).render(encoding=None))