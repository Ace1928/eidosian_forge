from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_modify_metadef_property(self):
    self.policy.modify_metadef_property()
    self.enforcer.enforce.assert_called_once_with(self.context, 'modify_metadef_property', mock.ANY)