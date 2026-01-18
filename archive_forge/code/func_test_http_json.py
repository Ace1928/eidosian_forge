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
def test_http_json(self):
    data = {'test': 'json_request'}
    path = '/v1/images'
    text = 'OK'
    self.mock.post(self.endpoint + path, text=text)
    headers = {'test': 'chunked_request'}
    resp, body = self.client.post(path, headers=headers, data=data)
    self.assertEqual(text, resp.text)
    self.assertIsInstance(self.mock.last_request.body, str)
    self.assertEqual(data, json.loads(self.mock.last_request.body))