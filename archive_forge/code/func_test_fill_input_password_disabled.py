import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_fill_input_password_disabled(self):
    html = HTML(u'<form><p>\n          <input type="password" name="pass" />\n        </p></form>') | HTMLFormFiller(data={'pass': 'bar'})
    self.assertEqual('<form><p>\n          <input type="password" name="pass"/>\n        </p></form>', html.render())