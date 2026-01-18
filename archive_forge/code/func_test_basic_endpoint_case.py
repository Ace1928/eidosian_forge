from testtools import matchers
from keystoneauth1.loading._plugins import admin_token as loader
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_basic_endpoint_case(self):
    self.stub_url('GET', ['p'], text='body')
    a = token_endpoint.Token(self.TEST_URL, self.TEST_TOKEN)
    s = session.Session(auth=a)
    data = s.get('/p', authenticated=True, endpoint_filter={'service': 'identity'})
    self.assertEqual(self.TEST_URL, a.get_endpoint(s))
    self.assertEqual('body', data.text)
    self.assertRequestHeaderEqual('X-Auth-Token', self.TEST_TOKEN)