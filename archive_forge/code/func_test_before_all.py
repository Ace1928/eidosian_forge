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
def test_before_all(self):
    self.assertEqual(self._apply('*|text()'), [(None, START, u'root'), (None, TEXT, u'CONTENT 1'), (OUTSIDE, TEXT, u'ROOT'), (None, TEXT, u'CONTENT 2'), (ENTER, START, u'foo'), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foo'), (None, TEXT, u'CONTENT 3'), (ENTER, START, u'bar'), (INSIDE, TEXT, u'BAR'), (EXIT, END, u'bar'), (None, END, u'root')])