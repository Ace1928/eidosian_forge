import os
from breezy.tests import TestCaseWithTransport
def test_missing_file(self):
    self.run_bzr_error(['Path\\(s\\) are not versioned: no-such-file'], 'inventory no-such-file')