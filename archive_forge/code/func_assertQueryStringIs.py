import json as jsonutils
import logging
import time
import urllib.parse
import uuid
import fixtures
import requests
from requests_mock.contrib import fixture
import testtools
def assertQueryStringIs(self, qs=''):
    """Verify the QueryString matches what is expected.

        The qs parameter should be of the format \\'foo=bar&abc=xyz\\'
        """
    expected = urllib.parse.parse_qs(qs, keep_blank_values=True)
    parts = urllib.parse.urlparse(self.requests_mock.last_request.url)
    querystring = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
    self.assertEqual(expected, querystring)