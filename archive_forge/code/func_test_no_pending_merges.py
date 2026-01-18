import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def test_no_pending_merges(self):
    self.run_bzr_error(['brz: ERROR: No pending merges present.\n'], ['rebase', '--pending-merges'])