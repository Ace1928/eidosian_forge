from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_add_image_falls_back_to_legacy(self):
    self.config(enforce_new_defaults=False, group='oslo_policy')
    self.config(enforce_scope=False, group='oslo_policy')
    self.context.is_admin = False
    self.policy = policy.ImageAPIPolicy(self.context, {'owner': 'else'}, enforcer=self.enforcer)
    self.assertRaises(exception.Forbidden, self.policy.add_image)
    with mock.patch('glance.api.v2.policy.check_admin_or_same_owner') as m:
        self.policy.add_image()
        m.assert_called_once_with(self.context, {'project_id': 'else', 'owner': 'else', 'visibility': 'private'})
    self.config(enforce_new_defaults=True, group='oslo_policy')
    self.config(enforce_scope=True, group='oslo_policy')
    with mock.patch('glance.api.v2.policy.check_admin_or_same_owner') as m:
        self.policy.add_image()
        m.assert_not_called()