import random
import string
from openstack import exceptions
from openstack.tests.functional import base
def test_get_role(self):
    role = self.operator_cloud.get_role('admin')
    self.assertIsNotNone(role)
    self.assertIn('id', role)
    self.assertIn('name', role)
    self.assertEqual('admin', role['name'])