import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_doctype_in_stream(self):
    stream = Stream([(Stream.DOCTYPE, DocType.HTML_STRICT, (None, -1, -1))])
    output = stream.render(XMLSerializer, encoding=None)
    self.assertEqual('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n', output)