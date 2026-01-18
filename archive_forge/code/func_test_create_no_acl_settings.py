from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_create_no_acl_settings(self):
    entity = self.manager.create(entity_ref=self.container_ref)
    self.assertEqual([], entity.operation_acls)
    self.assertEqual(self.container_ref, entity.entity_ref)
    self.assertEqual(self.container_ref + '/acl', entity.acl_ref)