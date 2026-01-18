import testtools
from unittest import mock
from troveclient.compat import auth
from troveclient.compat import exceptions
def test__v2_auth(self):
    username = 'trove_user'
    password = 'trove_password'
    tenant = 'tenant'
    cls_type = auth.KeyStoneV2Authenticator
    authObj = auth.KeyStoneV2Authenticator(url=None, type=cls_type, client=None, username=username, password=password, tenant=tenant)

    def side_effect_func(url, body):
        return body
    mock_obj = mock.Mock()
    mock_obj.side_effect = side_effect_func
    authObj._authenticate = mock_obj
    body = authObj._v2_auth(mock.Mock())
    self.assertEqual(username, body['auth']['passwordCredentials']['username'])
    self.assertEqual(password, body['auth']['passwordCredentials']['password'])
    self.assertEqual(tenant, body['auth']['tenantName'])