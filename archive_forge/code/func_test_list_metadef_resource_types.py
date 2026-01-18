from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_list_metadef_resource_types(self):
    self.policy.list_metadef_resource_types()
    self.enforcer.enforce.assert_called_once_with(self.context, 'list_metadef_resource_types', mock.ANY)