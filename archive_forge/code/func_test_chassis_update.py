from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_chassis_update(self):
    chassis = self.create_chassis()
    chassis.extra = {'answer': 42}
    chassis = self.conn.baremetal.update_chassis(chassis)
    self.assertEqual({'answer': 42}, chassis.extra)
    chassis = self.conn.baremetal.get_chassis(chassis.id)
    self.assertEqual({'answer': 42}, chassis.extra)