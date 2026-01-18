import uuid
from osc_placement.tests.functional import base
def test_list_non_key_value_resource_specified_error(self):
    self.assertCommandFailed('Arguments to --resource must be of form <resource_class>=<value>', self.openstack, 'allocation candidate list --resource VCPU')