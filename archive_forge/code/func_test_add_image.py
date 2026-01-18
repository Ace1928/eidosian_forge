from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_add_image(self):
    generic_target = {'project_id': self.context.project_id, 'owner': self.context.project_id, 'visibility': 'private'}
    self.policy = policy.ImageAPIPolicy(self.context, {}, enforcer=self.enforcer)
    self.policy.add_image()
    self.enforcer.enforce.assert_called_once_with(self.context, 'add_image', generic_target)