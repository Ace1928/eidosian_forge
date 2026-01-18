import uuid
from openstack import exceptions
from openstack.tests.functional import base
def test_find_flavors_by_name(self):
    rslt = self.conn.compute.find_flavor(self.one_flavor.name)
    self.assertEqual(rslt.name, self.one_flavor.name)