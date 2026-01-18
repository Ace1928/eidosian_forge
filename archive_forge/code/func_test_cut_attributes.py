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
def test_cut_attributes(self):
    self.assertEqual(self._apply('foo/@*', with_attrs=True), ([(None, START, (u'root', {})), (None, TEXT, u'ROOT'), (None, START, (u'foo', {})), (None, TEXT, u'FOO'), (None, END, u'foo'), (None, START, (u'bar', {u'name': u'bar'})), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')], [[(None, ATTR, {u'name': u'foo', u'size': u'100'})]]))