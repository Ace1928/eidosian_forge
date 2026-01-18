from openstack.compute.v2 import limits as _limits
from openstack.tests.functional import base
def test_get_our_volume_limits(self):
    """Test quotas functionality"""
    limits = self.user_cloud.get_volume_limits()
    self.assertIsNotNone(limits)
    self.assertFalse(hasattr(limits, 'maxTotalVolumes'))