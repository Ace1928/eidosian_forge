from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import hypervisors as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
def test_hypervisor_statistics(self):
    exc = self.assertRaises(exceptions.UnsupportedVersion, self.cs.hypervisor_stats.statistics)
    self.assertIn("The 'statistics' API is removed in API version 2.88 or later.", str(exc))