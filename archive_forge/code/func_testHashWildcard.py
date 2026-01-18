from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from gslib.commands import hash
from gslib.exception import CommandException
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
def testHashWildcard(self):
    num_test_files = 2
    tmp_dir = self.CreateTempDir(test_files=num_test_files)
    stdout = self.RunCommand('hash', args=[os.path.join(tmp_dir, '*')], return_stdout=True)
    num_expected_lines = num_test_files * (1 + 2)
    self.assertEqual(len(stdout.splitlines()), num_expected_lines)