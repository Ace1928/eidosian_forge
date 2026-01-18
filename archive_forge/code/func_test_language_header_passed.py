import functools
import json
import logging
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import session
from keystoneauth1 import token_endpoint
from oslo_utils import encodeutils
import requests
from requests_mock.contrib import fixture
from urllib import parse
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
import testtools
from testtools import matchers
import types
import glanceclient
from glanceclient.common import http
from glanceclient.tests import utils
def test_language_header_passed(self):
    kwargs = {'language_header': 'nb_NO'}
    http_client = http.HTTPClient(self.endpoint, **kwargs)
    path = '/v2/images/my-image'
    self.mock.get(self.endpoint + path)
    http_client.get(path)
    headers = self.mock.last_request.headers
    self.assertEqual(kwargs['language_header'], headers['Accept-Language'])