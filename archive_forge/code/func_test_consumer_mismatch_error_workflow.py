import base64
import uuid
import requests
from keystoneauth1 import exceptions
from keystoneauth1.extras import _saml2 as saml2
from keystoneauth1 import fixture as ksa_fixtures
from keystoneauth1 import session
from keystoneauth1.tests.unit.extras.saml2 import fixtures as saml2_fixtures
from keystoneauth1.tests.unit.extras.saml2 import utils
from keystoneauth1.tests.unit import matchers
def test_consumer_mismatch_error_workflow(self):
    consumer1 = 'http://keystone.test/Shibboleth.sso/SAML2/ECP'
    consumer2 = 'http://consumer2/Shibboleth.sso/SAML2/ECP'
    soap_response = saml2_fixtures.soap_response(consumer=consumer1)
    saml_assertion = saml2_fixtures.saml_assertion(destination=consumer2)
    self.requests_mock.get(self.default_sp_url, headers=CONTENT_TYPE_PAOS_HEADER, content=soap_response)
    self.requests_mock.post(self.TEST_IDP_URL, content=saml_assertion)
    saml_error = self.requests_mock.post(consumer1)
    self.assertRaises(exceptions.AuthorizationFailure, self.get_plugin().get_auth_ref, self.session)
    self.assertTrue(saml_error.called)