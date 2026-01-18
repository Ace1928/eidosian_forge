import unittest
from genshi.core import Attrs, QName, Stream
from genshi.input import XMLParser, HTMLParser, ParseError, ET
from genshi.compat import StringIO, BytesIO
from genshi.tests.test_utils import doctest_suite
from xml.etree import ElementTree
def test_html_entity_in_text(self):
    text = u'<p>&nbsp;</p>'
    events = list(HTMLParser(StringIO(text)))
    kind, data, pos = events[1]
    self.assertEqual(Stream.TEXT, kind)
    self.assertEqual(u'\xa0', data)