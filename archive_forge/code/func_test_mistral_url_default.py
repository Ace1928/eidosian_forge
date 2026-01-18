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
def test_mistral_url_default(self, http_client_mock, session_mock):
    session = mock.Mock()
    session_mock.side_effect = [session]
    get_endpoint = mock.Mock(side_effect=Exception)
    session.get_endpoint = get_endpoint
    client.client(username='mistral', project_name='mistral', api_key='password', user_domain_name='Default', project_domain_name='Default', auth_url=AUTH_HTTP_URL_v3)
    self.assertTrue(http_client_mock.called)
    mistral_url_for_http = http_client_mock.call_args[0][0]
    self.assertEqual(MISTRAL_HTTP_URL, mistral_url_for_http)