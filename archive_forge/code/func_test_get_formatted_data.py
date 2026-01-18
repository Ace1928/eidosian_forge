from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_get_formatted_data(self):
    s_entity = acls.SecretACL(api=None, entity_ref=self.secret_ref, users=self.users1)
    data = s_entity.read._get_formatted_data()
    self.assertEqual(acls.DEFAULT_OPERATION_TYPE, data[0])
    self.assertIsNone(data[1])
    self.assertEqual(self.users1, data[2])
    self.assertIsNone(data[3])
    self.assertIsNone(data[4])
    self.assertEqual(self.secret_acl_ref, data[5])
    c_entity = acls.ContainerACL(api=None, entity_ref=self.container_ref, users=self.users2, created=self.created)
    data = c_entity.get('read')._get_formatted_data()
    self.assertEqual(acls.DEFAULT_OPERATION_TYPE, data[0])
    self.assertIsNone(data[1])
    self.assertEqual(self.users2, data[2])
    self.assertEqual(timeutils.parse_isotime(self.created).isoformat(), data[3])
    self.assertIsNone(data[4])
    self.assertEqual(self.container_acl_ref, data[5])