import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_handles_encoding(self):
    self.create_branch()
    for encoding in self.good_encodings:
        self.try_encoding(encoding)