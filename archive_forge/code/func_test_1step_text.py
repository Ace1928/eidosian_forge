import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_1step_text(self):
    xml = XML('<root>Hey</root>')
    self._test_eval(path='text()', equiv='<Path "child::text()">', input=xml, output='Hey')
    self._test_eval(path='./text()', equiv='<Path "self::node()/child::text()">', input=xml, output='Hey')
    self._test_eval(path='//text()', equiv='<Path "descendant-or-self::text()">', input=xml, output='Hey')
    self._test_eval(path='.//text()', equiv='<Path "self::node()/descendant-or-self::node()/child::text()">', input=xml, output='Hey')