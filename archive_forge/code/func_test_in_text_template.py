import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_in_text_template(self):
    """
        Verify that the directive works as expected in a text template.
        """
    tmpl = TextTemplate("\n          #def echo(greeting, name='world')\n            ${greeting}, ${name}!\n          #end\n          ${echo('Hi', name='you')}\n        ")
    self.assertEqual('\n                      Hi, you!\n\n        ', tmpl.generate().render(encoding=None))