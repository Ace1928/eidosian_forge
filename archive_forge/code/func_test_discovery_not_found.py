import urllib
import uuid
import warnings
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import oidc
from keystoneauth1 import session
from keystoneauth1.tests.unit import oidc_fixtures
from keystoneauth1.tests.unit import utils
def test_discovery_not_found(self):
    self.requests_mock.get('http://not.found', status_code=404)
    plugin = self.plugin.__class__(self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, discovery_endpoint='http://not.found')
    self.assertRaises(exceptions.http.NotFound, plugin._get_discovery_document, self.session)