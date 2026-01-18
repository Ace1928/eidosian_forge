from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import hypervisors as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
def test_hypervisor_search_unicode(self):
    hypervisor_match = '\\u5de5\\u4f5c'
    if self.cs.api_version >= api_versions.APIVersion('2.53'):
        self.assertRaises(exceptions.BadRequest, self.cs.hypervisors.search, hypervisor_match)
    else:
        self.assertRaises(exceptions.NotFound, self.cs.hypervisors.search, hypervisor_match)