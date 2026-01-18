import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def assertUrlEqual(self, expected_url, provided_url):

    def parse_components(url):
        parsed = parse.urlsplit(url)
        query = parse.parse_qs(parsed.query)
        return (parsed._replace(query=''), query)
    expected_parse, expected_query = parse_components(expected_url)
    provided_parse, provided_query = parse_components(provided_url)
    self.assertEqual(expected_parse, provided_parse)
    self.assertEqual(expected_query, provided_query)