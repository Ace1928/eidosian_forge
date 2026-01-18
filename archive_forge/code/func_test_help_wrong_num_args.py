from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.command import Command
import gslib.tests.testcase as testcase
def test_help_wrong_num_args(self):
    stderr = self.RunGsUtil(['cp'], return_stderr=True, expected_status=1)
    self.assertIn('Usage:', stderr)