from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.command import Command
import gslib.tests.testcase as testcase
def test_help_noargs(self):
    stdout = self.RunCommand('help', return_stdout=True)
    self.assertIn('Available commands', stdout)