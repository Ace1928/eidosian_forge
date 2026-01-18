from keystoneauth1.loading._plugins import noauth as loader
from keystoneauth1 import noauth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_get_endpoint(self):
    a = noauth.NoAuth(endpoint=self.TEST_URL)
    s = session.Session(auth=a)
    self.assertEqual(self.TEST_URL, a.get_endpoint(s))