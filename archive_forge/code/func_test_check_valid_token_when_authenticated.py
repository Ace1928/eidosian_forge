import urllib
import uuid
from lxml import etree
from keystoneauth1 import exceptions
from keystoneauth1.extras import _saml2 as saml2
from keystoneauth1.tests.unit import client_fixtures
from keystoneauth1.tests.unit.extras.saml2 import fixtures as saml2_fixtures
from keystoneauth1.tests.unit.extras.saml2 import utils
from keystoneauth1.tests.unit import matchers
def test_check_valid_token_when_authenticated(self):
    self.requests_mock.register_uri('GET', self.FEDERATION_AUTH_URL, json=saml2_fixtures.UNSCOPED_TOKEN, headers=client_fixtures.AUTH_RESPONSE_HEADERS)
    self.session.session.cookies = [object()]
    self.adfsplugin._access_service_provider(self.session)
    response = self.adfsplugin.authenticated_response
    self.assertEqual(client_fixtures.AUTH_RESPONSE_HEADERS, response.headers)
    self.assertEqual(saml2_fixtures.UNSCOPED_TOKEN['token'], response.json()['token'])