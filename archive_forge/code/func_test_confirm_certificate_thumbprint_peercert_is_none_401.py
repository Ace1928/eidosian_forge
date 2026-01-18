import base64
import copy
import hashlib
import jwt.utils
import logging
import ssl
from testtools import matchers
import time
from unittest import mock
import uuid
import webob.dec
import fixtures
from oslo_config import cfg
import six
from six.moves import http_client
import testresources
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from keystonemiddleware.auth_token import _cache
from keystonemiddleware import external_oauth2_token
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit import client_fixtures
from keystonemiddleware.tests.unit import utils
def test_confirm_certificate_thumbprint_peercert_is_none_401(self):
    conf = copy.deepcopy(self._test_conf)
    self.set_middleware(conf=conf)

    def mock_resp(request, context):
        return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, access_token=self._token, active=True, cert_thumb=self._cert_thumb, metadata=self._default_metadata)
    self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
    self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
    resp = self.call_middleware(headers=get_authorization_header(self._token), expected_status=401, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})
    self.assertEqual(resp.headers.get('WWW-Authenticate'), 'Authorization OAuth 2.0 uri="%s"' % self._audience)