import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_fill_select_multi_value_auto(self):
    html = HTML(u'<form><p>\n          <select name="foo" multiple>\n            <option>1</option>\n            <option>2</option>\n            <option>3</option>\n          </select>\n        </p></form>') | HTMLFormFiller(data={'foo': ['1', '3']})
    self.assertEqual('<form><p>\n          <select name="foo" multiple="multiple">\n            <option selected="selected">1</option>\n            <option>2</option>\n            <option selected="selected">3</option>\n          </select>\n        </p></form>', html.render())