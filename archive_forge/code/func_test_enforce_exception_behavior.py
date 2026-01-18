from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_enforce_exception_behavior(self):
    with mock.patch.object(self.policy.enforcer, 'enforce') as mock_enf:
        self.policy.modify_member()
        self.assertTrue(mock_enf.called)
        mock_enf.reset_mock()
        mock_enf.side_effect = exception.Forbidden
        self.assertRaises(webob.exc.HTTPNotFound, self.policy.modify_member)
        mock_enf.assert_has_calls([mock.call(mock.ANY, 'get_image', mock.ANY)])
        mock_enf.reset_mock()
        mock_enf.side_effect = [lambda *a: None, exception.Forbidden]
        self.assertRaises(webob.exc.HTTPForbidden, self.policy.modify_member)
        mock_enf.assert_has_calls([mock.call(mock.ANY, 'get_image', mock.ANY), mock.call(mock.ANY, 'modify_member', mock.ANY)])