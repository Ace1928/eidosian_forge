import uuid
from osc_placement.tests.functional import base
def test_list_with_any_traits_old_microversion(self):
    groups = {'': {'resources': ('DISK_GB=1',), 'required': ('STORAGE_DISK_HDD,STORAGE_DISK_SSD',)}, '1': {'resources': ('VCPU=1',), 'required': ('HW_CPU_X86_AVX',)}}
    self.assertCommandFailed('Operation or argument is not supported with version 1.29', self.allocation_candidate_granular, groups=groups)