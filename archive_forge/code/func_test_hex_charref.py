import unittest
from genshi.core import Attrs, QName, Stream
from genshi.input import XMLParser, HTMLParser, ParseError, ET
from genshi.compat import StringIO, BytesIO
from genshi.tests.test_utils import doctest_suite
from xml.etree import ElementTree
def test_hex_charref(self):
    text = u'<span>&#x27;</span>'
    events = list(HTMLParser(StringIO(text)))
    self.assertEqual(3, len(events))
    self.assertEqual((Stream.START, ('span', ())), events[0][:2])
    self.assertEqual((Stream.TEXT, "'"), events[1][:2])
    self.assertEqual((Stream.END, 'span'), events[2][:2])