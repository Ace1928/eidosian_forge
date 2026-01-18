from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_get_metadef_object(self):
    self.policy.get_metadef_object()
    self.enforcer.enforce.assert_called_once_with(self.context, 'get_metadef_object', mock.ANY)