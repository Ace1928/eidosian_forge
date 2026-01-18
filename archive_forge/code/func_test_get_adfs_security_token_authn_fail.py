import urllib
import uuid
from lxml import etree
from keystoneauth1 import exceptions
from keystoneauth1.extras import _saml2 as saml2
from keystoneauth1.tests.unit import client_fixtures
from keystoneauth1.tests.unit.extras.saml2 import fixtures as saml2_fixtures
from keystoneauth1.tests.unit.extras.saml2 import utils
from keystoneauth1.tests.unit import matchers
def test_get_adfs_security_token_authn_fail(self):
    """Test proper parsing XML fault after bad authentication.

        An exceptions.AuthorizationFailure should be raised including
        error message from the XML message indicating where was the problem.
        """
    content = utils.make_oneline(self.ADFS_FAULT)
    self.requests_mock.register_uri('POST', self.IDENTITY_PROVIDER_URL, content=content, status_code=500)
    self.adfsplugin._prepare_adfs_request()
    self.assertRaises(exceptions.AuthorizationFailure, self.adfsplugin._get_adfs_security_token, self.session)