from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_deploy_template_negative_non_existing(self):
    uuid = 'bbb45f41-d4bc-4307-8d1d-32f95ce1e920'
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_deploy_template, uuid)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.delete_deploy_template, uuid, ignore_missing=False)
    self.assertIsNone(self.conn.baremetal.delete_deploy_template(uuid))