import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_long_simple_paths(self):
    xml = XML('<root><a><b><a><d><a><b><a><b><a><b><a><c>!</c></a></b></a></b></a></b></a></d></a></b></a></root>')
    self._test_eval('//a/b/a/b/a/c', input=xml, output='<c>!</c>')
    self._test_eval('//a/b/a/c', input=xml, output='<c>!</c>')
    self._test_eval('//a/c', input=xml, output='<c>!</c>')
    self._test_eval('//c', input=xml, output='<c>!</c>')
    self._test_eval('a/b/descendant::a/c', input=xml, output='<c>!</c>')
    self._test_eval('a/b/descendant::a/d/descendant::a/c', input=xml, output='<c>!</c>')
    self._test_eval('a/b/descendant::a/d/a/c', input=xml, output='')
    self._test_eval('//d/descendant::b/descendant::b/descendant::b/descendant::c', input=xml, output='<c>!</c>')
    self._test_eval('//d/descendant::b/descendant::b/descendant::b/descendant::b/descendant::c', input=xml, output='')