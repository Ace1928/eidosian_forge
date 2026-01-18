import json as jsonutils
import logging
import time
import urllib.parse
import uuid
import fixtures
import requests
from requests_mock.contrib import fixture
import testtools
def assertContentTypeIs(self, content_type):
    last_request = self.requests_mock.last_request
    self.assertEqual(last_request.headers['Content-Type'], content_type)