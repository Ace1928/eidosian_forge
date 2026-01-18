import unittest
from genshi.core import Attrs, QName, Stream
from genshi.input import XMLParser, HTMLParser, ParseError, ET
from genshi.compat import StringIO, BytesIO
from genshi.tests.test_utils import doctest_suite
from xml.etree import ElementTree
def test_element_attribute_order(self):
    text = '<elem title="baz" id="foo" class="bar" />'
    events = list(XMLParser(StringIO(text)))
    kind, data, pos = events[0]
    self.assertEqual(Stream.START, kind)
    tag, attrib = data
    self.assertEqual('elem', tag)
    self.assertEqual(('title', 'baz'), attrib[0])
    self.assertEqual(('id', 'foo'), attrib[1])
    self.assertEqual(('class', 'bar'), attrib[2])