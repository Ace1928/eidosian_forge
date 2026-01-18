import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_attrwildcard_with_namespace(self):
    xml = XML('<root xmlns:f="FOO"><foo f:bar="baz"/></root>')
    self._test_eval('foo[@f:*]', input=xml, output='<foo xmlns:ns1="FOO" ns1:bar="baz"/>', namespaces={'f': 'FOO'})