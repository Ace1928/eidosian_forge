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
def test_invert_inverted_element(self):
    self.assertEqual(_transform(FOO, Transformer('foo').invert().invert()), [(None, START, u'root'), (None, TEXT, u'ROOT'), (OUTSIDE, START, u'foo'), (OUTSIDE, TEXT, u'FOO'), (OUTSIDE, END, u'foo'), (None, END, u'root')])