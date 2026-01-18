from openstack import exceptions
from openstack.tests.functional import base
def test_delete_group_not_exists(self):
    self.assertFalse(self.operator_cloud.delete_group('xInvalidGroupx'))