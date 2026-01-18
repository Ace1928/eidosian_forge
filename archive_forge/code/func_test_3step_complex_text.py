import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_3step_complex_text(self):
    xml = XML('<root><item><bar>Some text </bar><baz><bar>in here.</bar></baz></item></root>')
    self._test_eval(path='item/bar/text()', equiv='<Path "child::item/child::bar/child::text()">', input=xml, output='Some text ')
    self._test_eval(path='item//bar/text()', equiv='<Path "child::item/descendant-or-self::node()/child::bar/child::text()">', input=xml, output='Some text in here.')