import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_2step(self):
    xml = XML('<root><foo/><bar/></root>')
    self._test_eval('*', input=xml, output='<foo/><bar/>')
    self._test_eval('bar', input=xml, output='<bar/>')
    self._test_eval('baz', input=xml, output='')