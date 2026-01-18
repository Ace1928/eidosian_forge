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
def test_get_access_token_failed_500_other(self):
    """Test case when unexpected error."""
    client_name = 'client_name_test'
    app_cred = self._create_app_cred(self.user_id, client_name)
    error = 'other_error'
    client_id = app_cred.get('id')
    client_secret = app_cred.get('secret')
    b64str = b64encode(f'{client_id}:{client_secret}'.encode()).decode().strip()
    headers = {'Content-Type': 'application/x-www-form-urlencoded', 'Authorization': f'Basic {b64str}'}
    data = {'grant_type': 'client_credentials'}
    data = parse.urlencode(data).encode()
    with mock.patch('keystone.api._shared.authentication.authenticate_for_token') as co_mock:
        co_mock.side_effect = exception.UnexpectedError('unexpected error.')
        resp = self.post(self.ACCESS_TOKEN_URL, headers=headers, convert=False, body=data, noauth=True, expected_status=client.INTERNAL_SERVER_ERROR)
    LOG.debug(f'response: {resp}')
    json_resp = jsonutils.loads(resp.body)
    self.assertEqual(error, json_resp.get('error'))