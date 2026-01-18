import pickle
import unittest
from genshi import core
from genshi.core import Markup, Attrs, Namespace, QName, escape, unescape
from genshi.input import XML
from genshi.compat import StringIO, BytesIO, IS_PYTHON2
from genshi.tests.test_utils import doctest_suite
def test_curly_brace_equality(self):
    qname1 = QName('{http://www.example.org/namespace}elem')
    qname2 = QName('http://www.example.org/namespace}elem')
    self.assertEqual(qname1.namespace, qname2.namespace)
    self.assertEqual(qname1.localname, qname2.localname)
    self.assertEqual(qname1, qname2)