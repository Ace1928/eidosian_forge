import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_fill_option_segmented_text_no_value(self):
    html = MarkupTemplate('<form>\n          <select name="foo">\n            <option>foo $x bar</option>\n          </select>\n        </form>').generate(x=1) | HTMLFormFiller(data={'foo': 'foo 1 bar'})
    self.assertEqual('<form>\n          <select name="foo">\n            <option selected="selected">foo 1 bar</option>\n          </select>\n        </form>', html.render())