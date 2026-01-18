import pickle
import unittest
from genshi import core
from genshi.core import Markup, Attrs, Namespace, QName, escape, unescape
from genshi.input import XML
from genshi.compat import StringIO, BytesIO, IS_PYTHON2
from genshi.tests.test_utils import doctest_suite
def test_leading_curly_brace(self):
    qname = QName('{http://www.example.org/namespace}elem')
    self.assertEqual('http://www.example.org/namespace', qname.namespace)
    self.assertEqual('elem', qname.localname)