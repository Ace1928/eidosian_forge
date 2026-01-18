import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def assert_parse_error_or_equal(self, expected, exploit, allow_strip=False):
    try:
        html = HTML(exploit)
    except ParseError:
        return
    sanitized_html = (html | HTMLSanitizer()).render()
    if not sanitized_html and allow_strip:
        return
    self.assertEqual(expected, sanitized_html)