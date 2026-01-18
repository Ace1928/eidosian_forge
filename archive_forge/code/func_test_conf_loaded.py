import uuid
from oslo_config import fixture as config
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_conf_loaded(self):
    token = uuid.uuid4().hex
    endpoint_filter = {'service_type': 'compute', 'service_name': 'nova', 'version': (2, 1)}
    loader = self.useLoadingFixture(token=token)
    url = loader.get_endpoint('/path', **endpoint_filter)
    m = self.requests_mock.get(url)
    auth = loading.load_auth_from_conf_options(self.conf_fixture.conf, self.GROUP)
    sess = session.Session(auth=auth)
    self.assertEqual(self.AUTH_TYPE, auth.auth_type)
    sess.get('/path', endpoint_filter=endpoint_filter)
    self.assertTrue(m.called_once)
    self.assertTrue(token, m.last_request.headers['X-Auth-Token'])
    self.assertEqual(loader.project_id, sess.get_project_id())
    self.assertEqual(loader.user_id, sess.get_user_id())