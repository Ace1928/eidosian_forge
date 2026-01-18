import os
from breezy.branch import Branch
from breezy.tests import TestCaseWithTransport
def check_modified(expected, null=False):
    command = 'modified'
    if null:
        command += ' --null'
    out, err = self.run_bzr(command)
    self.assertEqual(out, expected)
    self.assertEqual(err, '')