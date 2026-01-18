import os
import tempfile
from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from oslotest import base
import osprofiler.profiler
from mistralclient.api import client
@mock.patch('keystoneauth1.session.Session')
@mock.patch('mistralclient.api.httpclient.HTTPClient')
def test_mistral_profile_enabled(self, http_client_mock, session_mock):
    keystone_client_instance = self.setup_keystone_mock(session_mock)
    client.client(username='mistral', project_name='mistral', api_key='password', user_domain_name='Default', project_domain_name='Default', auth_url=AUTH_HTTP_URL_v3, profile=PROFILER_HMAC_KEY)
    self.assertTrue(http_client_mock.called)
    profiler = osprofiler.profiler.get()
    self.assertEqual(profiler.hmac_key, PROFILER_HMAC_KEY)