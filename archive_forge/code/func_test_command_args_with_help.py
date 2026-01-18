from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.command import Command
import gslib.tests.testcase as testcase
def test_command_args_with_help(self):
    stdout = self.RunCommand('cp', ['foo', 'bar', '--help'], return_stdout=True)
    self.assertIn('cp - Copy files and objects', stdout)