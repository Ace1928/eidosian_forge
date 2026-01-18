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
@mock.patch.object(session.Session, 'request')
def test_fetch_token_introspect_timeout_500(self, mock_session_request):
    conf = copy.deepcopy(self._test_conf)
    self.set_middleware(conf=conf)
    mock_session_request.side_effect = ksa_exceptions.RequestTimeout('time out')
    self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
    self.call_middleware(headers=get_authorization_header(self._token), expected_status=500, method='GET', path='/vnfpkgm/v1/vnf_packages', environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})