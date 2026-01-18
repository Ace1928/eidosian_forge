import pickle
import unittest
from genshi import core
from genshi.core import Markup, Attrs, Namespace, QName, escape, unescape
from genshi.input import XML
from genshi.compat import StringIO, BytesIO, IS_PYTHON2
from genshi.tests.test_utils import doctest_suite
def test_non_ascii(self):
    attrs_tuple = Attrs([('attr1', u'föö'), ('attr2', u'bär')]).totuple()
    self.assertEqual(u'fööbär', attrs_tuple[1])