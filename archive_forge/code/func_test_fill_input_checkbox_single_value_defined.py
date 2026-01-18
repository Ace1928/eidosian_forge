import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_fill_input_checkbox_single_value_defined(self):
    html = HTML('<form><p>\n          <input type="checkbox" name="foo" value="1" />\n        </p></form>', encoding='ascii')
    self.assertEqual('<form><p>\n          <input type="checkbox" name="foo" value="1" checked="checked"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': '1'})).render())
    self.assertEqual('<form><p>\n          <input type="checkbox" name="foo" value="1"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': '2'})).render())