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
def test_get_connections_kwargs_http(self):
    endpoint = 'http://example.com:9292'
    test_client = http.HTTPClient(endpoint, token='adc123')
    self.assertEqual(600.0, test_client.timeout)