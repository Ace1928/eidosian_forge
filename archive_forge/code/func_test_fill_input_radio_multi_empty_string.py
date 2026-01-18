import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_fill_input_radio_multi_empty_string(self):
    html = HTML(u'<form><p>\n          <input type="radio" name="foo" value="" />\n        </p></form>')
    self.assertEqual('<form><p>\n          <input type="radio" name="foo" value="" checked="checked"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': ['']})).render())