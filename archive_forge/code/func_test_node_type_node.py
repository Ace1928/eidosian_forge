import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_node_type_node(self):
    xml = XML('<root>Some text <br/>in here.</root>')
    self._test_eval(path='node()', equiv='<Path "child::node()">', input=xml, output='Some text <br/>in here.')