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
def test_attr_from_function(self):

    def set(name, event):
        self.assertEqual(name, 'name')
        return event[1][1].get('name').upper()
    self.assertEqual(self._attr('foo|bar', 'name', set), [(None, START, (u'root', {})), (None, TEXT, u'ROOT'), (ENTER, START, (u'foo', {u'name': 'FOO', u'size': '100'})), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foo'), (ENTER, START, (u'bar', {u'name': 'BAR'})), (INSIDE, TEXT, u'BAR'), (EXIT, END, u'bar'), (None, END, u'root')])