import pickle
import unittest
from genshi import core
from genshi.core import Markup, Attrs, Namespace, QName, escape, unescape
from genshi.input import XML
from genshi.compat import StringIO, BytesIO, IS_PYTHON2
from genshi.tests.test_utils import doctest_suite
def test_new_with_encoding(self):
    markup = Markup(u'Döner'.encode('utf-8'), encoding='utf-8')
    self.assertEqual('<Markup %r>' % u'Döner', repr(markup))