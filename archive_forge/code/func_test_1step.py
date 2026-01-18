import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_1step(self):
    xml = XML('<root><elem/></root>')
    self._test_eval(path='elem', equiv='<Path "child::elem">', input=xml, output='<elem/>')
    self._test_eval(path='elem', equiv='<Path "child::elem">', input=xml, output='<elem/>')
    self._test_eval(path='child::elem', equiv='<Path "child::elem">', input=xml, output='<elem/>')
    self._test_eval(path='//elem', equiv='<Path "descendant-or-self::elem">', input=xml, output='<elem/>')
    self._test_eval(path='descendant::elem', equiv='<Path "descendant::elem">', input=xml, output='<elem/>')