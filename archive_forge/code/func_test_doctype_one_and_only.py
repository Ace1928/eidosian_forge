import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_doctype_one_and_only(self):
    stream = Stream([(Stream.DOCTYPE, ('html', None, None), (None, -1, -1))])
    output = stream.render(XMLSerializer, doctype=DocType.HTML_STRICT, encoding=None)
    self.assertEqual('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n', output)