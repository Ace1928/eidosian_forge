import testtools
from openstack import exceptions
from openstack.tests.functional import base
def test_add_volume_type_access_missing_project(self):
    self.operator_cloud.add_volume_type_access('test-volume-type', '00000000000000000000000000000000')
    self.operator_cloud.remove_volume_type_access('test-volume-type', '00000000000000000000000000000000')