from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_manage_image_cache(self):
    self.policy = policy.CacheImageAPIPolicy(self.context, enforcer=self.enforcer, policy_str='manage_image_cache')
    self.policy.manage_image_cache()
    self.enforcer.enforce.assert_called_once_with(self.context, 'manage_image_cache', mock.ANY)