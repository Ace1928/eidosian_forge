import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_nested_defs(self):
    """
        Verify that a template function defined inside a conditional block can
        be called from outside that block.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:if test="semantic">\n            <strong py:def="echo(what)">${what}</strong>\n          </py:if>\n          <py:if test="not semantic">\n            <b py:def="echo(what)">${what}</b>\n          </py:if>\n          ${echo(\'foo\')}\n        </doc>')
    self.assertEqual('<doc>\n          <strong>foo</strong>\n        </doc>', tmpl.generate(semantic=True).render(encoding=None))