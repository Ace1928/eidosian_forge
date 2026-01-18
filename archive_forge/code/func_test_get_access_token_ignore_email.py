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
def test_get_access_token_ignore_email(self):
    """Test case when an access token can be successfully obtain."""
    self._create_mapping(dn_rules=[{'user.name': 'SSL_CLIENT_SUBJECT_DN_CN', 'user.id': 'SSL_CLIENT_SUBJECT_DN_UID', 'user.domain.id': 'SSL_CLIENT_SUBJECT_DN_DC', 'user.domain.name': 'SSL_CLIENT_SUBJECT_DN_O', 'SSL_CLIENT_ISSUER_DN_CN': ['root']}])
    user, user_domain, user_project = self._create_project_user()
    *_, client_cert, _ = self._create_certificates(client_dn=unit.create_dn(country_name='jp', state_or_province_name='kanagawa', locality_name='kawasaki', organizational_unit_name='test', user_id=user.get('id'), common_name=user.get('name'), domain_component=user_domain.get('id'), organization_name=user_domain.get('name')))
    cert_content = self._get_cert_content(client_cert)
    resp = self._get_access_token(client_id=user.get('id'), client_cert_content=cert_content)
    LOG.debug(resp)
    json_resp = jsonutils.loads(resp.body)
    self.assertIn('access_token', json_resp)
    self.assertEqual('Bearer', json_resp['token_type'])
    self.assertEqual(3600, json_resp['expires_in'])
    verify_resp = self.get('/auth/tokens', headers={'X-Subject-Token': json_resp['access_token'], 'X-Auth-Token': json_resp['access_token']})
    self.assertIn('token', verify_resp.result)
    self.assertIn('oauth2_credential', verify_resp.result['token'])
    self.assertIn('roles', verify_resp.result['token'])
    self.assertIn('project', verify_resp.result['token'])
    self.assertIn('catalog', verify_resp.result['token'])
    self.assertEqual(user_project.get('id'), verify_resp.result['token']['project']['id'])
    check_oauth2 = verify_resp.result['token']['oauth2_credential']
    self.assertEqual(utils.get_certificate_thumbprint(cert_content), check_oauth2['x5t#S256'])