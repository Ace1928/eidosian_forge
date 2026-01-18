import os
import re
import breezy
from breezy import ignores, osutils
from breezy.branch import Branch
from breezy.errors import CommandError
from breezy.osutils import pathjoin
from breezy.tests import TestCaseWithTransport
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_ignore_default_rules(self):
    out, err = self.run_bzr(['ignore', '--default-rules'])
    reference_set = set(ignores.USER_DEFAULTS)
    output_set = set(out.rstrip().split('\n'))
    self.assertEqual(reference_set, output_set)
    self.assertEqual('', err)