import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_nested_bound_namespaces(self):
    stream = Stream([(Stream.START_NS, ('x', 'http://example.org/'), (None, -1, -1)), (Stream.START, (QName('div'), Attrs()), (None, -1, -1)), (Stream.TEXT, '\n          ', (None, -1, -1)), (Stream.START_NS, ('x', 'http://example.org/'), (None, -1, -1)), (Stream.START, (QName('p'), Attrs()), (None, -1, -1)), (Stream.END, QName('p'), (None, -1, -1)), (Stream.END_NS, 'x', (None, -1, -1)), (Stream.TEXT, '\n          ', (None, -1, -1)), (Stream.START_NS, ('x', 'http://example.org/'), (None, -1, -1)), (Stream.START, (QName('p'), Attrs()), (None, -1, -1)), (Stream.END, QName('p'), (None, -1, -1)), (Stream.END_NS, 'x', (None, -1, -1)), (Stream.TEXT, '\n        ', (None, -1, -1)), (Stream.END, QName('div'), (None, -1, -1)), (Stream.END_NS, 'x', (None, -1, -1))])
    output = stream.render(XHTMLSerializer, encoding=None)
    self.assertEqual('<div xmlns:x="http://example.org/">\n          <p></p>\n          <p></p>\n        </div>', output)