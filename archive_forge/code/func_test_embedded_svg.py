import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_embedded_svg(self):
    text = '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:svg="http://www.w3.org/2000/svg">\n          <body>\n            <button>\n              <svg:svg width="600px" height="400px">\n                <svg:polygon id="triangle" points="50,50 50,300 300,300"></svg:polygon>\n              </svg:svg>\n            </button>\n          </body>\n        </html>'
    output = XML(text).render(XHTMLSerializer, encoding=None)
    self.assertEqual(text, output)