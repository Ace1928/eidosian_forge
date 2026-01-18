import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_node_type_comment(self):
    xml = XML('<root><!-- commented --></root>')
    self._test_eval(path='comment()', equiv='<Path "child::comment()">', input=xml, output='<!-- commented -->')