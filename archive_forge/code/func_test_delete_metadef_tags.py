from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_delete_metadef_tags(self):
    self.policy.delete_metadef_tags()
    self.enforcer.enforce.assert_called_once_with(self.context, 'delete_metadef_tags', mock.ANY)