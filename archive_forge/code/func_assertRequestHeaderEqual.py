import json as jsonutils
import logging
import time
import urllib.parse
import uuid
import fixtures
import requests
from requests_mock.contrib import fixture
import testtools
def assertRequestHeaderEqual(self, name, val):
    """Verify that the last request made contains a header and its value.

        The request must have already been made.
        """
    headers = self.requests_mock.last_request.headers
    self.assertEqual(headers.get(name), val)