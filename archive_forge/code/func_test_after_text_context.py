import doctest
import unittest
import six
from genshi import HTML
from genshi.builder import Element
from genshi.compat import IS_PYTHON2
from genshi.core import START, END, TEXT, QName, Attrs
from genshi.filters.transform import Transformer, StreamBuffer, ENTER, EXIT, \
import genshi.filters.transform
from genshi.tests.test_utils import doctest_suite
def test_after_text_context(self):
    self.assertEqual(self._apply('.', html='foo'), [(OUTSIDE, TEXT, u'foo'), (None, TEXT, u'CONTENT 1')])