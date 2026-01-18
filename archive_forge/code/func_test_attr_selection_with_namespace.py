import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_attr_selection_with_namespace(self):
    xml = XML('<root xmlns:ns1="http://example.com"><foo ns1:bar="abc"></foo></root>')
    path = Path('foo/@ns1:bar')
    result = path.select(xml, namespaces={'ns1': 'http://example.com'})
    self.assertEqual(list(result), [Attrs([(QName('http://example.com}bar'), u'abc')])])