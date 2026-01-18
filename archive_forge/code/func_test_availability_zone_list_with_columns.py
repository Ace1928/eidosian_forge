import ddt
from oslo_utils import uuidutils
from manilaclient.tests.functional import base
@ddt.data(('name', ['Name']), ('name,id', ['Name', 'Id']), ('name,created_at', ['Name', 'Created_At']), ('name,id,created_at', ['Name', 'Id', 'Created_At']))
@ddt.unpack
def test_availability_zone_list_with_columns(self, columns_arg, expected):
    azs = self.user_client.list_availability_zones(columns=columns_arg)
    for az in azs:
        self.assertEqual(len(expected), len(az))
        for key in expected:
            self.assertIn(key, az)
    if 'Id' in expected:
        self.assertTrue(uuidutils.is_uuid_like(az['Id']))
    if 'Name' in expected:
        self.assertIsNotNone(az['Name'])
    if 'Created_At' in expected:
        self.assertIsNotNone(az['Created_At'])