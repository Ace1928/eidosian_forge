import datetime
from fixtures import TimeoutException
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_get_compute_usage(self):
    """Test usage functionality"""
    if not self.operator_cloud:
        self.skipTest('Operator cloud is required for this test')
    self.addCleanup(self._cleanup_servers_and_volumes, self.server_name)
    self.user_cloud.create_server(name=self.server_name, image=self.image, flavor=self.flavor, wait=True)
    start = datetime.datetime.now() - datetime.timedelta(seconds=5)
    usage = self.operator_cloud.get_compute_usage('demo', start)
    self.add_info_on_exception('usage', usage)
    self.assertIsNotNone(usage)
    self.assertIn('total_hours', usage)
    self.assertIn('start', usage)
    self.assertEqual(start.isoformat(), usage['start'])
    self.assertIn('location', usage)