from base64 import b64encode
from cryptography.hazmat.primitives.serialization import Encoding
import fixtures
import http
from http import client
from oslo_log import log
from oslo_serialization import jsonutils
from unittest import mock
from urllib import parse
from keystone.api.os_oauth2 import AccessTokenResource
from keystone.common import provider_api
from keystone.common import utils
from keystone import conf
from keystone import exception
from keystone.federation.utils import RuleProcessor
from keystone.tests import unit
from keystone.tests.unit import test_v3
from keystone.token.provider import Manager
def test_get_access_token_method_get_not_allowed(self):
    """Test case when the request is get method that is not allowed."""
    client_name = 'client_name_test'
    app_cred = self._create_app_cred(self.user_id, client_name)
    json_resp = self._get_access_token_method_not_allowed(app_cred, self.get)
    self.assertEqual('other_error', json_resp.get('error'))
    self.assertEqual('The method is not allowed for the requested URL.', json_resp.get('error_description'))