import urllib
import uuid
import warnings
from keystoneauth1 import exceptions
from keystoneauth1.identity.v3 import oidc
from keystoneauth1 import session
from keystoneauth1.tests.unit import oidc_fixtures
from keystoneauth1.tests.unit import utils
def test_can_pass_grant_type_but_warning_is_issued(self):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        self.plugin.__class__(self.AUTH_URL, self.IDENTITY_PROVIDER, self.PROTOCOL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET, grant_type=self.GRANT_TYPE)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert 'grant_type' in str(w[-1].message)