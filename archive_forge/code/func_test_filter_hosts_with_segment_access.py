from neutron_lib.plugins.ml2 import api
from neutron_lib.tests import _base as base
def test_filter_hosts_with_segment_access(self):
    dummy_token = ['X']
    self.assertEqual(dummy_token, _MechanismDriver().filter_hosts_with_segment_access(dummy_token, dummy_token, dummy_token, dummy_token))