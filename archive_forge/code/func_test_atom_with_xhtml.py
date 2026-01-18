import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_atom_with_xhtml(self):
    text = '<feed xmlns="http://www.w3.org/2005/Atom" xml:lang="en">\n            <id>urn:uuid:c60843aa-0da8-4fa6-bbe5-98007bc6774e</id>\n            <updated>2007-01-28T11:36:02.807108-06:00</updated>\n            <title type="xhtml">\n                <div xmlns="http://www.w3.org/1999/xhtml">Example</div>\n            </title>\n            <subtitle type="xhtml">\n                <div xmlns="http://www.w3.org/1999/xhtml">Bla bla bla</div>\n            </subtitle>\n            <icon/>\n        </feed>'
    output = XML(text).render(XMLSerializer, encoding=None)
    self.assertEqual(text, output)