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
def test_logs_warning(self):
    req = webob.Request.blank('/')
    self.middleware(req.environ, self.start_fake_response)
    self.assertEqual(self.response_status, 200)
    log = 'Use of the auth_uri option was deprecated in the Queens release in favor of www_authenticate_uri.'
    self.assertThat(self.logger.output, matchers.Contains(log))