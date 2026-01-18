from openstack.block_storage.v2 import limits
from openstack.tests.unit import base
def test_make_limit(self):
    limit_resource = limits.Limit(**LIMIT)
    self._test_rate_limits(LIMIT['rate'], limit_resource.rate)
    self._test_absolute_limit(LIMIT['absolute'], limit_resource.absolute)