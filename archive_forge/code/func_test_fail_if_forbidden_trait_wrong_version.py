import operator
import uuid
from osc_placement.tests.functional import base
def test_fail_if_forbidden_trait_wrong_version(self):
    self.assertCommandFailed('Operation or argument is not supported with version 1.18', self.resource_provider_list, resources=('MEMORY_MB=1024', 'DISK_GB=80'), forbidden=('STORAGE_DISK_HDD',))