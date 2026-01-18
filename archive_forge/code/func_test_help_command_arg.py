from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.command import Command
import gslib.tests.testcase as testcase
def test_help_command_arg(self):
    stdout = self.RunCommand('help', ['ls'], return_stdout=True)
    self.assertIn('ls - List providers, buckets', stdout)