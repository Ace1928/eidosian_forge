from openstack.tests.functional import base
def test_magnum_services(self):
    """Test magnum services functionality"""
    services = self.operator_cloud.list_magnum_services()
    self.assertEqual(1, len(services))
    self.assertEqual(services[0]['id'], 1)
    self.assertEqual('up', services[0]['state'])
    self.assertEqual('magnum-conductor', services[0]['binary'])
    self.assertGreater(services[0]['report_count'], 0)