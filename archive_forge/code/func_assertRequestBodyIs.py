import json as jsonutils
import logging
import time
import urllib.parse
import uuid
import fixtures
import requests
from requests_mock.contrib import fixture
import testtools
def assertRequestBodyIs(self, body=None, json=None):
    last_request_body = self.requests_mock.last_request.body
    if json:
        val = jsonutils.loads(last_request_body)
        self.assertEqual(json, val)
    elif body:
        self.assertEqual(body, last_request_body)