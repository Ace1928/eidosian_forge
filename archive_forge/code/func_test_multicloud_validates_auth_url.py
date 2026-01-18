from unittest import mock
import webob
from webob import exc
from heat.common import auth_url
from heat.tests import common
@mock.patch.object(auth_url.AuthUrlFilter, '_validate_auth_url')
@mock.patch.object(auth_url.cfg, 'CONF')
def test_multicloud_validates_auth_url(self, mock_cfg, mock_validate):
    mock_cfg.auth_password.multi_cloud = True
    req = webob.Request.blank('/tenant_id/')
    self.middleware(req)
    self.assertTrue(mock_validate.called)