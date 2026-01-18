from openstack.database.v1 import user
from openstack.tests.unit import base
def test_make(self):
    sot = user.User(**CREATING)
    self.assertEqual(CREATING['name'], sot.id)
    self.assertEqual(CREATING['databases'], sot.databases)
    self.assertEqual(CREATING['name'], sot.name)
    self.assertEqual(CREATING['name'], sot.id)
    self.assertEqual(CREATING['password'], sot.password)