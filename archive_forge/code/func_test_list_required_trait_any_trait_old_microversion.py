import operator
import uuid
from osc_placement.tests.functional import base
def test_list_required_trait_any_trait_old_microversion(self):
    self.assertCommandFailed('Operation or argument is not supported with version 1.22', self.resource_provider_list, resources=('MEMORY_MB=1024', 'DISK_GB=80'), required=('STORAGE_DISK_HDD,STORAGE_DISK_SSD', 'HW_NIC_SRIOV_MULTIQUEUE'))