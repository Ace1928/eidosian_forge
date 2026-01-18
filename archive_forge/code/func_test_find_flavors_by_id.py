import uuid
from openstack import exceptions
from openstack.tests.functional import base
def test_find_flavors_by_id(self):
    rslt = self.conn.compute.find_flavor(self.one_flavor.id)
    self.assertEqual(rslt.id, self.one_flavor.id)