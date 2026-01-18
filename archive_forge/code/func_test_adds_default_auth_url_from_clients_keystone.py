from unittest import mock
import webob
from webob import exc
from heat.common import auth_url
from heat.tests import common
@mock.patch.object(auth_url.cfg, 'CONF')
def test_adds_default_auth_url_from_clients_keystone(self, mock_cfg):
    self.config = {}
    mock_cfg.clients_keystone.auth_uri = 'foobar'
    mock_cfg.keystone_authtoken.www_authenticate_uri = 'should-be-ignored'
    mock_cfg.auth_password.multi_cloud = False
    with mock.patch('keystoneauth1.discover.Discover') as discover:

        class MockDiscover(object):

            def url_for(self, endpoint):
                return 'foobar/v3'
        discover.return_value = MockDiscover()
        self.middleware = auth_url.AuthUrlFilter(self.app, self.config)
        req = webob.Request.blank('/tenant_id/')
        self.middleware(req)
        self.assertIn('X-Auth-Url', req.headers)
        self.assertEqual('foobar/v3', req.headers['X-Auth-Url'])