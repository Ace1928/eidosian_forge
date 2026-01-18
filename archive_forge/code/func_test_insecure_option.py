from unittest import mock
import urllib.parse
import fixtures
from oslo_serialization import jsonutils
import requests
from requests_mock.contrib import fixture as rm_fixture
from testtools import matchers
import webob
from keystonemiddleware import s3_token
from keystonemiddleware.tests.unit import utils
def test_insecure_option(self):
    true_values = ['true', 'True', '1', 'yes']
    for val in true_values:
        config = {'insecure': val, 'certfile': 'false_ind'}
        middleware = s3_token.filter_factory(config)(FakeApp())
        self.assertIs(False, middleware._verify)
    false_values = ['false', 'False', '0', 'no', 'someweirdvalue']
    for val in false_values:
        config = {'insecure': val, 'certfile': 'false_ind'}
        middleware = s3_token.filter_factory(config)(FakeApp())
        self.assertEqual('false_ind', middleware._verify)
    config = {'certfile': 'false_ind'}
    middleware = s3_token.filter_factory(config)(FakeApp())
    self.assertIs('false_ind', middleware._verify)