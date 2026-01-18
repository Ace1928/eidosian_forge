import unittest
from genshi.core import Attrs, QName, Stream
from genshi.input import XMLParser, HTMLParser, ParseError, ET
from genshi.compat import StringIO, BytesIO
from genshi.tests.test_utils import doctest_suite
from xml.etree import ElementTree
def test_html_entity_with_dtd(self):
    text = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"\n        "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n        <html>&nbsp;</html>\n        '
    events = list(XMLParser(StringIO(text)))
    kind, data, pos = events[2]
    self.assertEqual(Stream.TEXT, kind)
    self.assertEqual(u'\xa0', data)