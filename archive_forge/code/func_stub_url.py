import json as jsonutils
import logging
import time
import urllib.parse
import uuid
import fixtures
import requests
from requests_mock.contrib import fixture
import testtools
def stub_url(self, method, parts=None, base_url=None, json=None, **kwargs):
    if not base_url:
        base_url = self.TEST_URL
    if json:
        kwargs['text'] = jsonutils.dumps(json)
        headers = kwargs.setdefault('headers', {})
        headers.setdefault('Content-Type', 'application/json')
    if parts:
        url = '/'.join([p.strip('/') for p in [base_url] + parts])
    else:
        url = base_url
    url = url.replace('/?', '?')
    return self.requests_mock.register_uri(method, url, **kwargs)