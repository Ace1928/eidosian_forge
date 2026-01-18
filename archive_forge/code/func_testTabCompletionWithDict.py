from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import fire
from fire import test_components as tc
from fire import testutils
import mock
import six
def testTabCompletionWithDict(self):
    actions = {'multiply': lambda a, b: a * b}
    completion_script = fire.Fire(actions, command=['--', '--completion'], name='actCLI')
    self.assertIn('actCLI', completion_script)
    self.assertIn('multiply', completion_script)