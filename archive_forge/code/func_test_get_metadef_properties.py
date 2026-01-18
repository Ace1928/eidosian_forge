from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_get_metadef_properties(self):
    self.policy.get_metadef_properties()
    self.enforcer.enforce.assert_called_once_with(self.context, 'get_metadef_properties', mock.ANY)