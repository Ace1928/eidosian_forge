from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_chassis_patch(self):
    chassis = self.create_chassis()
    chassis = self.conn.baremetal.patch_chassis(chassis, dict(path='/extra/answer', op='add', value=42))
    self.assertEqual({'answer': 42}, chassis.extra)
    chassis = self.conn.baremetal.get_chassis(chassis.id)
    self.assertEqual({'answer': 42}, chassis.extra)