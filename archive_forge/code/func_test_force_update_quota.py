from novaclient import api_versions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import quotas as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
def test_force_update_quota(self):
    q = super(QuotaSetsTest2_57, self).test_force_update_quota()
    for invalid_resource in self.invalid_resources:
        self.assertFalse(hasattr(q, invalid_resource), '%s should not be in %s' % (invalid_resource, q))