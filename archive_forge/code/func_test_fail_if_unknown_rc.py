import uuid
from osc_placement.tests.functional import base
def test_fail_if_unknown_rc(self):
    self.assertCommandFailed('No such resource', self.allocation_candidate_list, resources=('UNKNOWN=10',))