import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_html5_doctype(self):
    stream = HTML(u'<html></html>')
    output = stream.render(HTMLSerializer, doctype=DocType.HTML5, encoding=None)
    self.assertEqual('<!DOCTYPE html>\n<html></html>', output)