import unittest
from genshi.core import Attrs, QName, Stream
from genshi.input import XMLParser, HTMLParser, ParseError, ET
from genshi.compat import StringIO, BytesIO
from genshi.tests.test_utils import doctest_suite
from xml.etree import ElementTree
def test_html_entity_in_attribute(self):
    text = u'<p title="&nbsp;"></p>'
    events = list(HTMLParser(StringIO(text)))
    kind, data, pos = events[0]
    self.assertEqual(Stream.START, kind)
    self.assertEqual(u'\xa0', data[1].get('title'))
    kind, data, pos = events[1]
    self.assertEqual(Stream.END, kind)