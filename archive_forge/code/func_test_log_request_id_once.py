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
def test_log_request_id_once(self):
    logger = self.useFixture(fixtures.FakeLogger(level=logging.DEBUG))
    data = 'TEST'
    path = '/v1/images/'
    self.mock.get(self.endpoint + path, body=io.StringIO(data), headers={'Content-Type': 'application/octet-stream', 'x-openstack-request-id': '1234'})
    resp, body = self.client.get(path)
    self.assertIsInstance(body, types.GeneratorType)
    self.assertEqual([data], list(body))
    expected_log = 'GET call to image for http://example.com:9292/v1/images/ used request id 1234'
    self.assertEqual(1, logger.output.count(expected_log))