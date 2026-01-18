import uuid
from osc_placement.tests.functional import base
def test_fail_if_aggregate_uuid_wrong_version(self):
    self.assertCommandFailed('Operation or argument is not supported with version 1.17', self.allocation_candidate_list, resources=('MEMORY_MB=1024', 'DISK_GB=80'), aggregate_uuids=[str(uuid.uuid4())])
    self.assertCommandFailed('Operation or argument is not supported with version 1.17', self.allocation_candidate_list, resources=('MEMORY_MB=1024', 'DISK_GB=80'), member_of=[str(uuid.uuid4())])